"""
nlr_meltinglevels.py

Haalt de hoogte van het 0 graden C-niveau (freezing level) en het -20 graden C-niveau
op voor een gegeven locatie en tijdstip, via de gratis Open-Meteo API (open-meteo.com).

Bedoeld gebruik in NLradar: per radarbestand (lat/lon van de radar, datetime van de scan)
de smeltlaag- en -20C-hoogte opvragen, bijvoorbeeld voor HCLASS-klasse-restricties
(vergelijkbaar met de melting-layer-beperkingen in de NEXRAD HCA van Park et al. 2009).

Databron en betrouwbaarheid (GEVERIFIEERD via handmatige tests op 22 juli 2026)
--------------------------------------------------------------------------------
- 0C-hoogte komt rechtstreeks van Open-Meteo's variabele 'freezing_level_height'.
- -20C-hoogte bestaat niet als kant-en-klare variabele en wordt afgeleid uit de
  temperatuur-/geopotentiele-hoogte-reeksen op vaste drukniveaus (1000..30 hPa),
  via lineaire interpolatie naar het punt waar de temperatuur -20C kruist.
- BELANGRIJKE BEPERKING, empirisch vastgesteld: de Historical Forecast API
  (api.open-meteo.com/v1/forecast met start_date in het verleden) levert
  drukniveaus alleen betrouwbaar voor ongeveer de laatste 2 weken. Bij tests
  werkte dit nog op 14 dagen terug, was leeg op 21 dagen terug, en gaf de
  endpoint zelfs een HTTP 400-fout voorbij ~2-4 maanden terug.
  'freezing_level_height' (dus de 0C-hoogte) bleef in diezelfde tests wel
  beschikbaar tot minstens 60 dagen terug - de 0C-hoogte is dus aanzienlijk
  robuuster dan de -20C-afleiding.
- De ERA5-archiefroute (archive-api.open-meteo.com/v1/archive), oorspronkelijk
  bedoeld als fallback voor data van voor 2021, biedt HELEMAAL GEEN
  drukniveau-variabelen (bevestigd via de officiele Open-Meteo-documentatie
  van die endpoint: het aanbod is beperkt tot oppervlakte-/bodemvariabelen).
  Deze route levert dus nooit een -20C-hoogte, ongeacht de datum.
  Conclusie: voor een -20C-hoogte van radardata ouder dan ~2-3 weken heeft
  Open-Meteo simpelweg geen dekking. Voor de 0C-hoogte is de dekking merkbaar
  ruimer, maar ook niet onbeperkt (ergens tussen 60 dagen en enkele maanden
  terug houdt ook die op).
- Om dit te ondervangen is er een lokaal, PERSISTENT archief (sqlite-bestand
  'melting_level_archive.sqlite3', naast dit modulebestand): elke succesvolle
  opvraag wordt daar blijvend in bewaard. Als NLradar dus actueel wordt gebruikt
  (binnen het venster waarin Open-Meteo de data nog heeft), staat de waarde
  daarna permanent lokaal vast en hoeft er nooit meer een nieuwe API-aanroep
  voor dat exacte radar/uur te gebeuren - ook niet als je maanden later
  hetzelfde volume opnieuw bekijkt. Voor volumes die voor het EERST worden
  bekeken nadat het Open-Meteo-venster al gesloten is, blijft de data
  onbeschikbaar (h0_m/h_minus20_m worden dan None); daar is met de huidige
  aanpak niets aan te doen.
- Voor recente/actuele data wordt de Historical Forecast API gebruikt. LET OP
  (empirisch vastgesteld): het model 'knmi_seamless' bleek GEEN van beide
  variabelen te leveren (KNMI Harmonie-AROME ondersteunt 'freezing_level_height'
  domeinbreed niet - hourly_units toonde 'undefined' voor die variabele).
  Standaardmodel is daarom None (Open-Meteo's automatische "best match"), wat
  in tests wel werkte (freezing_level_height en drukniveaus beide gevuld).
  'dwd_icon' werkte in tests ook.
- Geen API-key nodig; gratis voor niet-commercieel gebruik (CC BY 4.0). Geen
  aanwijzing gevonden dat een key de bovenstaande beperkingen zou opheffen -
  de ontbrekende drukniveaus in het ERA5-archief zijn een documentatiematig
  bevestigd ontbrekend aanbod, geen quota- of toegangsprobleem.

Belangrijk: dit bestand is in de sandbox NIET tegen de echte Open-Meteo-API
getest (netwerktoegang tot open-meteo.com is hier geblokkeerd) - de bevindingen
hierboven komen uit handmatige tests die Erik zelf heeft uitgevoerd. De
interpolatielogica is wel apart getest met synthetische data (zie onderaan
dit bestand, if __name__ == '__main__').
"""

import datetime as _dt
import json
import os
import re
import socket
import sqlite3
import ssl
import urllib.request
import urllib.parse
import urllib.error


# Vaste drukniveaus die Open-Meteo aanbiedt (hPa), aflopend qua druk = oplopend qua hoogte.
PRESSURE_LEVELS_HPA = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500,
                       400, 300, 250, 200, 150, 100, 70, 50, 30]

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Vanaf deze datum heeft de Historical Forecast API dekking (modelspecifieke archieven).
_HISTORICAL_FORECAST_START = _dt.date(2021, 1, 1)

# Simpele in-memory cache: voorkomt herhaalde calls voor dezelfde scan binnen 1 sessie.
_cache = {}

# Persistent lokaal archief: staat altijd naast dit modulebestand, dus onafhankelijk
# van waar NLradar vanuit gestart wordt. Elke succesvolle opvraag wordt hier
# blijvend bewaard, zodat data die je NU (binnen Open-Meteo's beschikbaarheidsvenster)
# opvraagt, ook nog lokaal beschikbaar is als je hetzelfde volume maanden later
# opnieuw bekijkt, ver buiten dat venster.
_ARCHIVE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "melting_level_archive.sqlite3")
# Installeerbare versie: programmamap is alleen-lezen -> archief in %LOCALAPPDATA%\NLradar\Generated_files
if getattr(__import__('sys'), 'frozen', False):
    _ARCHIVE_DB_PATH = os.path.join(os.environ.get('LOCALAPPDATA') or os.path.expanduser('~'),
                                    'NLradar', 'Generated_files', 'melting_level_archive.sqlite3')
    os.makedirs(os.path.dirname(_ARCHIVE_DB_PATH), exist_ok=True)


def _archive_connect():
    conn = sqlite3.connect(_ARCHIVE_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS melting_levels (
            cache_key TEXT PRIMARY KEY,
            h0_m REAL,
            h_minus20_m REAL,
            source TEXT,
            model TEXT,
            fetched_at_utc TEXT
        )
    """)
    # BUGFIX (23 juli, na Eriks melding dat een vers opgehaald resultaat bij hernieuwd bezoek
    # weer "--" gaf voor de datum, zelfs met een gloednieuw, leeg sqlite-bestand): de tabel
    # hierboven had NOOIT een datetime_used-kolom, en _archive_set/_archive_get sloegen dat veld
    # dus stilzwijgend nooit op/terug - een vers resultaat toonde de datum wel (rechtstreeks uit
    # het in-memory-dict), maar zodra hetzelfde punt de TWEEDE keer uit het archief werd gelezen,
    # was het veld al onderweg verdwenen. "CREATE TABLE IF NOT EXISTS" voegt geen kolom toe aan
    # een tabel die al bestaat (zoals bij Erik, na maanden testen), dus hier een expliciete
    # migratie: probeer de kolom toe te voegen, en negeer de fout als hij er al is (nieuwe,
    # eerder vandaag met deze fix aangemaakte archieven hebben 'm al via CREATE TABLE hierboven
    # niet - vandaar deze aparte ALTER TABLE, die voor zowel oude als gloednieuwe bestanden werkt).
    try:
        conn.execute("ALTER TABLE melting_levels ADD COLUMN datetime_used TEXT")
    except sqlite3.OperationalError:
        pass  # kolom bestaat al
    return conn


def _archive_get(cache_key_str):
    """Zoek een eerder opgeslagen resultaat op in het lokale archief. Retourneert
    de dict {'h0_m', 'h_minus20_m', 'source', 'model', 'datetime_used'} of None als niet aanwezig.
    """
    try:
        conn = _archive_connect()
        try:
            cur = conn.execute(
                "SELECT h0_m, h_minus20_m, source, model, datetime_used FROM melting_levels WHERE cache_key = ?",
                (cache_key_str,))
            row = cur.fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        # Een archiefprobleem (bv. corrupt bestand) mag nooit de opvraag zelf blokkeren -
        # dan valt dit gewoon terug op een live API-aanroep.
        return None
    if row is None:
        return None
    return {"h0_m": row[0], "h_minus20_m": row[1], "source": row[2], "model": row[3], "datetime_used": row[4]}


def _archive_set(cache_key_str, result):
    """Sla een resultaat blijvend op in het lokale archief. Fouten hierbij worden
    stilzwijgend genegeerd (het live-resultaat is dan al teruggegeven aan de aanroeper;
    alleen het voor-de-toekomst-bewaren mislukt, wat geen reden is om de aanroep zelf
    te laten falen).
    """
    try:
        conn = _archive_connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO melting_levels
                   (cache_key, h0_m, h_minus20_m, source, model, fetched_at_utc, datetime_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (cache_key_str, result["h0_m"], result["h_minus20_m"], result["source"],
                 result["model"], _dt.datetime.utcnow().isoformat(), result.get("datetime_used")))
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:
        pass


def archive_entry_count():
    """Handig om te checken hoeveel volumes er al lokaal gearchiveerd zijn."""
    try:
        conn = _archive_connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM melting_levels").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


class MeltingLevelError(Exception):
    """Fout bij het ophalen of interpreteren van de melting-level-data."""
    pass


def _round_to_hour(dt_utc):
    """Rond een datetime af naar het dichtstbijzijnde uur (Open-Meteo levert per uur)."""
    if dt_utc.minute >= 30:
        dt_utc = dt_utc + _dt.timedelta(hours=1)
    return dt_utc.replace(minute=0, second=0, microsecond=0)


def _http_get_json(url, params, timeout=15):
    full_url = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(full_url, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise MeltingLevelError(
            "Open-Meteo HTTP-fout {}: {} (url: {})".format(e.code, body, full_url))
    except urllib.error.URLError as e:
        raise MeltingLevelError(
            "Kan Open-Meteo niet bereiken: {} (url: {})".format(e.reason, full_url))
    except Exception as e:
        raise MeltingLevelError(
            "Onverwachte fout bij ophalen Open-Meteo-data: {} (url: {})".format(e, full_url))

    if isinstance(data, dict) and data.get("error"):
        raise MeltingLevelError(
            "Open-Meteo meldde een fout: {}".format(data.get("reason", "onbekend")))
    return data


def _interpolate_minus20_height(temps_c, heights_m):
    """
    Zoek de hoogte waar de temperatuur -20C kruist, via lineaire interpolatie
    tussen de twee omliggende drukniveaus.

    temps_c, heights_m: lijsten van gelijke lengte, oplopend in hoogte
    (dus van hoge druk/laag naar lage druk/hoog).

    Retourneert de hoogte in meter, of None als -20C niet binnen het
    beschikbare bereik ligt (bv. te warm/te koud profiel, of ontbrekende data).
    """
    return _interpolate_temp_crossing_height(temps_c, heights_m, -20.0)


def _interpolate_temp_crossing_height(temps_c, heights_m, target_temp):
    """Algemene versie van _interpolate_minus20_height hierboven: zoek de hoogte waar de
    temperatuur een WILLEKEURIG doel kruist (i.p.v. altijd -20C), via dezelfde lineaire
    interpolatie tussen de twee omliggende niveaus. Toegevoegd (22 juli) voor de Wyoming-
    sounding-fallback hieronder, die zowel het 0C- als het -20C-niveau uit dezelfde
    temperatuur/hoogte-kolommen moet halen.
    """
    n = len(temps_c)
    if n != len(heights_m) or n < 2:
        return None

    # Filter None/NaN-waarden eruit maar behoud volgorde.
    pairs = [(t, h) for t, h in zip(temps_c, heights_m)
             if t is not None and h is not None]
    if len(pairs) < 2:
        return None

    for i in range(len(pairs) - 1):
        t0, h0 = pairs[i]
        t1, h1 = pairs[i + 1]
        d0 = t0 - target_temp
        d1 = t1 - target_temp
        if d0 == 0:
            return h0
        if (d0 > 0) != (d1 > 0):
            frac = d0 / (t0 - t1) if (t0 - t1) != 0 else 0.0
            return h0 + frac * (h1 - h0)
    return None


def _select_endpoint_and_params(lat, lon, dt_utc, model):
    date_str = dt_utc.date().isoformat()
    pressure_vars = []
    for lvl in PRESSURE_LEVELS_HPA:
        pressure_vars.append("temperature_{}hPa".format(lvl))
        pressure_vars.append("geopotential_height_{}hPa".format(lvl))

    hourly_vars = ["freezing_level_height"] + pressure_vars

    if dt_utc.date() >= _HISTORICAL_FORECAST_START:
        url = FORECAST_URL
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(hourly_vars),
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "UTC",
        }
        if model:
            params["models"] = model
    else:
        # Te oud voor de Historical Forecast API -> ERA5-reanalyse.
        url = ARCHIVE_URL
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(hourly_vars),
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "UTC",
        }
    return url, params


WYOMING_STATION_DE_BILT = "06260"    # WMO-nummer De Bilt - alleen 00 UTC
WYOMING_STATION_MEPPEN = "10304"     # WMO-nummer Meppen, Duitsland (vlak over de grens) - 00+12 UTC
# UITBREIDING (23 juli, Eriks eigen vondst): nog 2 Duitse stations vlak over de grens, met
# hetzelfde 00/12 UTC-schema als Meppen - puur als extra achtervang/redundantie (ze liggen qua
# TIJD nooit dichterbij dan Meppen zelf, aangezien alle drie exact dezelfde twee lanceertijden
# hebben), voor als Meppen voor een specifieke dag toevallig geen bruikbare data heeft.
WYOMING_STATION_ESSEN = "10410"      # WMO-nummer Essen, Duitsland - 00+12 UTC
WYOMING_STATION_NORDERNEY = "10113"  # WMO-nummer Norderney, Duitsland (Waddeneilanden) - 00+12 UTC


def _parse_wyoming_sounding_text(text):
    """Haalt de PRES/HGHT/TEMP-kolommen uit de platte-tekst-respons van de Wyoming-
    sounding-URL (geen HTML, direct een tabel met een streepjeslijn erboven/-onder).
    Retourneert een lijst van (hoogte_m, temperatuur_C)-tupels."""
    rows = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r"^-?\d+\.\d+\s+-?\d+", stripped):
            velden = stripped.split()
            if len(velden) >= 3:
                try:
                    hght, temp = float(velden[1]), float(velden[2])
                    rows.append((hght, temp))
                except ValueError:
                    continue
    return rows


# Losse, kleine cache specifiek voor de Wyoming-opvraag (22 juli, performance-fix): deze
# bron geeft sowieso maar 1 waarde voor heel Nederland, dus is compleet onafhankelijk van
# lat/lon - een volledige roosterupdate (get_melting_levels_grid, 16 punten) zou zonder deze
# cache tot 16x hetzelfde, blokkerende netwerkverzoek doen als Open-Meteo voor alle 16 punten
# op dezelfde manier faalt (bv. omdat de datum te oud is). Erik meldde hierdoor meerdere
# vastlopers van 12-13 seconden ("t_derived_tot") na het inbouwen van de Wyoming-fallback -
# exact het symptoom van 16 sequentiele, blokkerende HTTP-aanvragen in de hoofdthread.
_wyoming_cache = {}


def _get_melting_levels_from_wyoming(dt_rounded):
    """Terugval-bron voor als Open-Meteo niets (meer) teruggeeft voor dit tijdstip (te oud
    voor zowel de forecast- als de archive-API - drukniveau-gegevens zijn daar maar ~2-3
    weken terug beschikbaar). Gebruikt het University of Wyoming radiosonde-archief: echte
    ballonmetingen, decennia terug beschikbaar.

    LET OP - reeele beperkingen, niet instelbare details:
    - Geeft 1 vaste waarde voor heel Nederland (geen rooster van 16 punten zoals de
      Open-Meteo-weg in get_melting_levels_grid hieronder). Vandaar de losse
      _wyoming_cache hierboven: deze functie hoeft dus maar 1x per tijdstip het netwerk op,
      ongeacht voor hoeveel roosterpunten ze wordt aangeroepen.
    - Alleen bruikbaar voor 00 of 12 UTC (radiosondes gaan niet elk uur omhoog, in
      tegenstelling tot Open-Meteo's uurlijkse modeldata). Kiest altijd het qua tijd
      dichtstbijzijnde beschikbare sondemoment (De Bilt 00Z, of Meppen 00Z/12Z), OOK over
      dag-grenzen heen - een scan om 22:00 UTC kan dus de 00:00 UTC-sounding van de VOLGENDE
      dag gebruiken als die dichterbij ligt dan alles van dezelfde dag (23 juli, na Eriks
      concrete voorbeeld: 27 juni 22:00 UTC koos eerst ten onrechte De Bilt 00Z VAN DIEZELFDE
      DAG, 22 uur eerder, terwijl Meppen 12Z - 10 uur eerder - of de volgende ochtend - 2 uur
      later - dichterbij liggen).
    - De Bilt (06260) lanceert alleen om 00 UTC. Meppen (10304, net over de Duitse grens,
      op Eriks eigen suggestie toegevoegd) lanceert ook om 12 UTC.

    Bevestigd werkend door Erik (22 juli 2026) op 4 echte datums: De Bilt voor 2024-07-23,
    2026-07-13 en 2026-03-01 (allemaal 00 UTC), en Meppen voor 2026-07-10 (12 UTC).

    Raises
    ------
    MeltingLevelError
        Als ook deze bron geen bruikbare data teruggeeft (bv. station tijdelijk niet
        beschikbaar, of daadwerkelijk geen sounding op dat tijdstip). Een eerder MISLUKTE
        poging voor hetzelfde tijdstip wordt NIET gecachet (in tegenstelling tot een
        geslaagde) - anders zou 1 tijdelijke netwerkhapering alle 16 roosterpunten voor de
        rest van de sessie blijven blokkeren.
    """
    # HERZIEN (23 juli, tweede keer, op Eriks verzoek): de vorige versie koos alleen tussen
    # "dezelfde dag 00Z" (De Bilt) en "dezelfde dag 12Z" (Meppen), puur op basis van dag/nacht.
    # Erik wees terecht op een concreet, beter voorbeeld: voor 27 juni 22:00 UTC koos die
    # logica De Bilt 00Z VAN DIEZELFDE DAG (22 uur eerder) - terwijl Meppen 12Z (10 uur eerder)
    # dichterbij ligt, en de 00Z-sounding van de VOLGENDE dag (28 juni, maar 2 uur later) qua
    # tijd zelfs het dichtstbij zou zijn. De oude logica keek nooit over dag-grenzen heen.
    # Nieuwe aanpak: bouw ALLE kandidaat-sondemomenten (De Bilt 00Z en Meppen 00Z/12Z, voor
    # de dag ervoor/dezelfde dag/de dag erna - een sounding kan hooguit ~36 uur van het
    # gevraagde tijdstip af liggen bij dit venster), bereken voor elk het werkelijke tijdsverschil
    # in uren, en sorteer op kleinste verschil - dat wordt als eerste geprobeerd, de rest als
    # noodgreep in oplopende volgorde. Zo wordt altijd het qua tijd meest logische moment als
    # eerste gebruikt, ongeacht dag-grenzen.
    # UITBREIDING (23 juli, Eriks eigen vondst): Essen en Norderney hebben hetzelfde 00/12
    # UTC-schema als Meppen - toegevoegd als extra achtervang/redundantie. Bij een exacte
    # gelijke tijdsafstand (kan voorkomen, want deze 3 Duitse stations delen precies dezelfde
    # lanceertijden) bepaalt een vaste voorkeursvolgorde (Meppen voor Essen voor Norderney) de
    # volgorde - Meppen ligt van de drie het dichtst bij het centrum van Nederland.
    STATION_VOORKEUR = {WYOMING_STATION_DE_BILT: 0, WYOMING_STATION_MEPPEN: 1,
                        WYOMING_STATION_ESSEN: 2, WYOMING_STATION_NORDERNEY: 3}
    kandidaten = []
    for dag_offset in (-1, 0, 1):
        kandidaat_dag = dt_rounded + _dt.timedelta(days=dag_offset)
        for station, uur in ((WYOMING_STATION_DE_BILT, 0),
                              (WYOMING_STATION_MEPPEN, 0),
                              (WYOMING_STATION_MEPPEN, 12),
                              (WYOMING_STATION_ESSEN, 0),
                              (WYOMING_STATION_ESSEN, 12),
                              (WYOMING_STATION_NORDERNEY, 0),
                              (WYOMING_STATION_NORDERNEY, 12)):
            kandidaat_dt = _dt.datetime(kandidaat_dag.year, kandidaat_dag.month, kandidaat_dag.day, uur)
            verschil_uren = abs((kandidaat_dt - dt_rounded).total_seconds()) / 3600.
            datetime_str = "{:04d}-{:02d}-{:02d}%20{}:00:00".format(
                kandidaat_dag.year, kandidaat_dag.month, kandidaat_dag.day, uur)
            kandidaten.append((verschil_uren, STATION_VOORKEUR[station], station, datetime_str, kandidaat_dt))
    kandidaten.sort(key=lambda k: (k[0], k[1]))
    pogingen = [(station, datetime_str, kandidaat_dt) for (_, _, station, datetime_str, kandidaat_dt) in kandidaten]

    laatste_fout = None
    for station, datetime_str, kandidaat_dt in pogingen:
        result, fout = _fetch_wyoming_single(station, datetime_str, kandidaat_dt)
        if fout is not None:
            laatste_fout = fout
        if result is not None:
            return result

    raise MeltingLevelError(
        "Wyoming-sounding-archief gaf voor geen van de 4 stations (De Bilt, Meppen, Essen, "
        "Norderney) "
        "bruikbare data voor {}: {}".format(dt_rounded, laatste_fout))


def _fetch_wyoming_single(station, datetime_str, kandidaat_dt):
    """Haalt de sounding op voor EEN specifiek station+tijdstip - gedeelde kernlogica,
    losgetrokken (25 juli, voor de handmatige terugvaloptie hieronder) uit de automatische
    kandidatenlus in _get_melting_levels_from_wyoming hierboven. Gebruikt dezelfde in-memory-
    en permanente-archiefcache, dus voordeel van beide paden samen (een automatisch gevonden
    resultaat hoeft niet opnieuw opgehaald te worden als je hetzelfde station/tijdstip later
    handmatig kiest, en andersom).

    Retourneert (result, fout): result is None als dit specifieke station+tijdstip niets
    bruikbaars opleverde (fout bevat dan de laatst opgetreden exceptie, of None als het
    gewoon een lege/te korte sounding was); anders is result de resultaat-dict en is fout None.
    """
    wyoming_cache_key = (station, datetime_str)
    wyoming_archive_key_str = "wyoming:{}:{}".format(station, datetime_str)

    if wyoming_cache_key in _wyoming_cache:
        cached = dict(_wyoming_cache[wyoming_cache_key])
        cached["from_cache"] = "memory"
        return cached, None

    # BUGFIX (22 juli, "wil niet verder terug dan 22:22:18Z"): de in-memory-cache
    # hierboven werkt prima BINNEN 1 draaiende sessie, maar gaat verloren bij elke
    # herstart van NLradar. Vandaar OOK een permanente (sqlite-)cache, met een eigen
    # sleutel-voorvoegsel ("wyoming:...") zodat die niet botst met de gewone
    # per-(lat,lon,uur)-cache van get_melting_levels hieronder.
    archived = _archive_get(wyoming_archive_key_str)
    if archived is not None:
        archived = dict(archived)
        archived["from_cache"] = "archive"
        _wyoming_cache[wyoming_cache_key] = archived
        return archived, None

    text = None
    laatste_fout = None
    for src in ("UNKNOWN", "BUFR"):
        url = ("https://weather.uwyo.edu/wsgi/sounding?datetime={}&id={}&src={}&type=TEXT:LIST"
               .format(datetime_str, station, src))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            break
        except (urllib.error.URLError, socket.timeout) as e:
            # BUGFIX (23 juli, na Eriks crash-log): een timeout die pas optreedt TIJDENS het
            # lezen van de respons (na een al geslaagde handshake) komt bij urllib soms als
            # een rauwe socket.timeout naar buiten, NIET als urllib.error.URLError - dat
            # ving deze except eerst niet op, waardoor zo'n trage/onderbroken verbinding als
            # onafgevangen crash naar boven lekte i.p.v. netjes naar het volgende station
            # door te schakelen. Vandaar hier ook socket.timeout meegevangen.
            laatste_fout = e
            # LET OP - VEILIGHEIDSKANTTEKENING (22 juli, na Eriks SSL-foutmelding: "self
            # signed certificate in certificate chain"): dit duidt op een lokaal
            # netwerk/antivirus/firewall dat HTTPS-verkeer onderschept met een eigen
            # certificaat - de browser vertrouwt dat (via Windows' certificaatwinkel),
            # maar Python's eigen, meegeleverde certificatenlijst niet. Als specifiek DIT
            # (certificaatverificatie) de oorzaak is, wordt hier EENMALIG opnieuw
            # geprobeerd ZONDER certificaatverificatie - alleen voor deze ene, publieke,
            # alleen-lezen meteorologische bron (geen gevoelige data). Dit verlaagt de
            # beveiliging van DEZE ENE verbinding - aanvaardbaar voor dit doel, maar
            # bewust niet toegepast op Open-Meteo, die dit probleem niet had. De "juiste"
            # oplossing is het onderliggende certificaat in Python's eigen
            # certificatenlijst te krijgen (bv. via pip-system-certs).
            if isinstance(e, urllib.error.URLError) and (
                    isinstance(e.reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(e)):
                try:
                    onveilige_context = ssl.create_default_context()
                    onveilige_context.check_hostname = False
                    onveilige_context.verify_mode = ssl.CERT_NONE
                    with urllib.request.urlopen(req, timeout=5, context=onveilige_context) as resp:
                        text = resp.read().decode("utf-8", errors="replace")
                    break
                except (urllib.error.URLError, socket.timeout) as e2:
                    laatste_fout = e2
                    continue
            continue

    if text is None:
        return None, laatste_fout

    rows = _parse_wyoming_sounding_text(text)
    if len(rows) < 2:
        return None, laatste_fout

    heights_m = [r[0] for r in rows]
    temps_c = [r[1] for r in rows]
    h0 = _interpolate_temp_crossing_height(temps_c, heights_m, 0.0)
    h_minus20 = _interpolate_temp_crossing_height(temps_c, heights_m, -20.0)

    result = {
        "h0_m": h0,
        "h_minus20_m": h_minus20,
        "source": "wyoming_sounding",
        "model": "station_{}".format(station),
        # ZICHTBAARHEID (23 juli, op Eriks verzoek): het exacte tijdstip van deze sounding
        # (kan flink afwijken van de scantijd, ook qua DAG - een scan om 22:00 UTC kan bv.
        # de 00:00 UTC-sounding van de VOLGENDE dag gebruiken, als die het dichtstbij ligt).
        "datetime_used": kandidaat_dt.isoformat(),
        "from_cache": "live",
    }
    _wyoming_cache[wyoming_cache_key] = result
    if h_minus20 is not None:
        # Alleen een bruikbaar resultaat permanent archiveren (zelfde principe als bij
        # de gewone get_melting_levels-cache) - een resultaat zonder bruikbare
        # -20C-waarde zou anders voor altijd "niets" blijven teruggeven voor deze
        # dag+station.
        _archive_set(wyoming_archive_key_str, result)
    return result, None


# Naam -> WMO-stationcode, voor de handmatige terugvaloptie hieronder (GUI kan deze namen
# rechtstreeks in een dropdown tonen).
WYOMING_STATIONS = {
    "De Bilt": WYOMING_STATION_DE_BILT,
    "Meppen": WYOMING_STATION_MEPPEN,
    "Essen": WYOMING_STATION_ESSEN,
    "Norderney": WYOMING_STATION_NORDERNEY,
}


def get_melting_levels_wyoming_manual(station, year, month, day, hour):
    """HANDMATIGE TERUGVALOPTIE (25 juli, op Eriks verzoek): haal de 0C-/-20C-hoogte rechtstreeks
    op voor EEN door de gebruiker gekozen station+tijdstip, in plaats van de automatische
    dichtstbijzijnde-tijd-keuze in _get_melting_levels_from_wyoming.

    Dit is uitdrukkelijk een TERUGVALOPTIE naast de automatische keuze, geen vervanging: de
    automatische keuze blijft de standaard in get_melting_levels(). De GUI kan deze functie
    apart aanroepen (bv. via een dropdown/dialoogvenster met station+datum+uur) wanneer Erik
    de automatische keuze wil overrulen.

    Parameters
    ----------
    station : str
        Stationsnaam uit WYOMING_STATIONS ("De Bilt", "Meppen", "Essen", "Norderney"),
        of rechtstreeks een van de WYOMING_STATION_*-WMO-codes.
    year, month, day : int
        De gewenste sondedatum (UTC).
    hour : int
        0 of 12 - de enige twee lanceertijden die deze bron kent. De Bilt lanceert alleen
        om 0 UTC (zie ValueError hieronder als 12 wordt opgegeven voor De Bilt).

    Returns
    -------
    dict, zelfde vorm als get_melting_levels() (h0_m, h_minus20_m, source, model,
    datetime_used, from_cache).

    Raises
    ------
    ValueError
        Bij een ongeldig uur (niet 0 of 12), of 12 UTC voor De Bilt (bestaat niet).
    MeltingLevelError
        Als dit specifieke station+tijdstip geen bruikbare sounding oplevert. In tegenstelling
        tot de automatische functie wordt hier NIET verder gezocht naar een ander station/uur -
        de gebruiker koos dit bewust.
    """
    station_code = WYOMING_STATIONS.get(station, station)
    if hour not in (0, 12):
        raise ValueError("Uur moet 0 of 12 UTC zijn, kreeg: {}".format(hour))
    if station_code == WYOMING_STATION_DE_BILT and hour != 0:
        raise ValueError("De Bilt lanceert alleen om 0 UTC, niet om {} UTC".format(hour))

    kandidaat_dt = _dt.datetime(year, month, day, hour)
    datetime_str = "{:04d}-{:02d}-{:02d}%20{}:00:00".format(year, month, day, hour)

    result, fout = _fetch_wyoming_single(station_code, datetime_str, kandidaat_dt)
    if result is None:
        raise MeltingLevelError(
            "Geen bruikbare Wyoming-sounding voor station {} op {}: {}".format(
                station, datetime_str, fout))
    return result


def get_melting_levels(lat, lon, dt_utc, model=None, use_cache=True):
    """
    Haal de 0C- en -20C-hoogte (in meter boven zeeniveau) op voor een locatie en tijdstip.

    Parameters
    ----------
    lat, lon : float
        WGS84-coordinaten van de radar (of het scangebied).
    dt_utc : datetime.datetime
        Tijdstip van de radarscan, in UTC (naive of tz-aware; wordt als UTC behandeld).
    model : str or None
        Open-Meteo modelnaam. LET OP (empirisch vastgesteld): 'knmi_seamless' bleek
        GEEN van beide variabelen te leveren (KNMI Harmonie-AROME ondersteunt
        'freezing_level_height' domeinbreed niet). Standaard daarom None (Open-Meteo's
        automatische "best match"), wat in tests wel werkte. 'dwd_icon' werkte ook.
        Wordt genegeerd voor data ouder dan 2021 (dan wordt altijd ERA5 gebruikt,
        wat overigens sowieso geen drukniveaus levert, zie boven).
    use_cache : bool
        Hergebruik een eerder opgehaald resultaat voor hetzelfde (afgeronde) uur/locatie/model -
        eerst uit het in-memory geheugen van dit proces, anders uit het lokale, permanente
        sqlite-archief (melting_level_archive.sqlite3, naast dit bestand). Een succesvolle
        live-opvraag wordt na afloop altijd in dat archief weggeschreven.

    Returns
    -------
    dict met:
        'h0_m'       : hoogte van het 0C-niveau in meter (float of None)
        'h_minus20_m': hoogte van het -20C-niveau in meter (float of None)
        'source'     : 'forecast_api' of 'archive_api' (welk endpoint gebruikt is)
        'model'      : het model dat is gebruikt, of None bij automatische keuze

    Raises
    ------
    MeltingLevelError
        Bij netwerkfouten, HTTP-fouten, of onverwachte/ontbrekende data.
    """
    if dt_utc.tzinfo is not None:
        dt_utc = dt_utc.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    dt_rounded = _round_to_hour(dt_utc)
    cache_key = (round(lat, 3), round(lon, 3), dt_rounded.isoformat(), model)
    cache_key_str = repr(cache_key)

    if use_cache and cache_key in _cache:
        cached_result = dict(_cache[cache_key])
        cached_result["from_cache"] = "memory"
        return cached_result

    if use_cache:
        archived = _archive_get(cache_key_str)
        if archived is not None:
            archived = dict(archived)
            archived["from_cache"] = "archive"
            _cache[cache_key] = archived
            return archived

    url, params = _select_endpoint_and_params(lat, lon, dt_rounded, model)

    try:
        data = _http_get_json(url, params)

        try:
            hourly = data["hourly"]
            times = hourly["time"]
        except KeyError as e:
            raise MeltingLevelError("Onverwacht antwoordformaat van Open-Meteo (mist '{}')".format(e))

        target_iso = dt_rounded.strftime("%Y-%m-%dT%H:%M")
        try:
            idx = times.index(target_iso)
        except ValueError:
            raise MeltingLevelError(
                "Tijdstip {} niet gevonden in Open-Meteo-antwoord (beschikbaar: {}..{})".format(
                    target_iso, times[0] if times else "?", times[-1] if times else "?"))

        h0 = hourly.get("freezing_level_height", [None] * len(times))[idx]

        temps_c = []
        heights_m = []
        for lvl in PRESSURE_LEVELS_HPA:
            t = hourly.get("temperature_{}hPa".format(lvl), [None] * len(times))[idx]
            h = hourly.get("geopotential_height_{}hPa".format(lvl), [None] * len(times))[idx]
            temps_c.append(t)
            heights_m.append(h)
        # PRESSURE_LEVELS_HPA loopt van hoge naar lage druk = van laag naar hoog in de atmosfeer,
        # dus de volgorde is al oplopend in hoogte - geen sortering nodig.

        h_minus20 = _interpolate_minus20_height(temps_c, heights_m)

        result = {
            "h0_m": h0,
            "h_minus20_m": h_minus20,
            "source": "forecast_api" if url == FORECAST_URL else "archive_api",
            "model": model if url == FORECAST_URL else "era5",
            # ZICHTBAARHEID (23 juli, op Eriks verzoek): het exacte tijdstip WAARVOOR deze
            # temperatuurdata geldt - bij Open-Meteo is dat gewoon hetzelfde (afgeronde) uur
            # als de scan zelf, want Open-Meteo is uurlijks. Bij de Wyoming-fallback
            # hieronder is dit WEL relevant om te tonen, want dat kan een heel ander tijdstip
            # zijn dan de scan (bv. een scan om 15:00 UTC gebruikt de 12:00 UTC-sounding).
            "datetime_used": dt_rounded.isoformat(),
            "from_cache": "live",
        }

        if h_minus20 is None:
            # BUGFIX (22 juli, na Eriks melding dat MESH op 27 juni niets liet zien): Open-
            # Meteo geeft voor tijdstippen buiten het ~2-3-wekenvenster voor drukniveau-
            # gegevens GEEN foutmelding - de aanvraag "slaagt" gewoon (geldig JSON-antwoord,
            # 'hourly'/'time' aanwezig), maar de temperature_XXXhPa/geopotential_height_XXXhPa-
            # velden zijn dan leeg (None). Zonder deze check werd dat stilzwijgend als
            # "succesvol resultaat" met h_minus20_m=None doorgegeven, en werd de Wyoming-
            # terugval hieronder (die alleen op een EXCEPTIE reageert) dus nooit aangeroepen.
            # Hier daarom ook proberen als Open-Meteo's -20C-waarde None is, niet alleen bij
            # een echte fout. Als de Wyoming-poging zelf ook niets oplevert, blijft het oude,
            # milde gedrag behouden (dict met h_minus20_m=None teruggeven, geen crash) - dat
            # is nodig omdat HCLASS bewust ZONDER temperatuur kan classificeren en dus een
            # dict met None verwacht, niet een uitzondering.
            try:
                wyoming_result = _get_melting_levels_from_wyoming(dt_rounded)
                if wyoming_result.get("h_minus20_m") is not None:
                    result = wyoming_result
            except MeltingLevelError:
                pass
    except MeltingLevelError:
        # Open-Meteo had niets (meer) voor dit tijdstip (meestal: te oud, buiten het
        # ~2-3-weken-venster voor drukniveau-gegevens) - terugval op het University of
        # Wyoming radiosonde-archief (22 juli 2026, op Eriks verzoek toegevoegd; zie
        # _get_melting_levels_from_wyoming hierboven voor de bekende beperkingen: 1 vast
        # punt i.p.v. rooster, alleen 00/12 UTC). Als ook dit faalt, komt de
        # MeltingLevelError van DIE functie gewoon naar boven - geen stille tweede fout.
        result = _get_melting_levels_from_wyoming(dt_rounded)

    if use_cache:
        _cache[cache_key] = result
        # BUGFIX (22 juli, na Eriks melding dat 27 juni nog steeds niets gaf ondanks de
        # Wyoming-fallback): het permanente sqlite-archief bewaarde voorheen OOK resultaten
        # met h_minus20_m=None voor altijd - een tijdstip dat ooit (voor deze fix, of tijdens
        # een tijdelijke storing) niets opleverde, bleef daardoor voorgoed "niets" teruggeven,
        # zelfs nadat de code verbeterd was of de bron weer beschikbaar kwam. Vanaf nu wordt
        # alleen een resultaat met een bruikbare h_minus20_m permanent gearchiveerd; een
        # "leeg" resultaat blijft wel in het (tijdelijke) in-memory-cache van dit proces, om
        # herhaalde verzoeken binnen dezelfde sessie te voorkomen, maar wordt niet voor altijd
        # vastgezet in het archief.
        if result.get("h_minus20_m") is not None:
            _archive_set(cache_key_str, result)

    return result


def get_melting_levels_grid(grid_points, dt_utc, model=None, use_cache=True):
    """Haalt de 0C- en -20C-hoogte op voor een LIJST van (lat, lon)-punten (bv. nlr_hclass.NL_GRID_POINTS),
    t.b.v. een gedeeld temperatuurrooster i.p.v. 1 vast punt per radar (zie gesprek met Erik, 22 juli).

    Roept simpelweg get_melting_levels() aan voor elk punt - profiteert dus van dezelfde in-memory- en
    lokale-archiefcache: eenmaal opgehaald per punt/uur/model, daarna gratis bij hernieuwd gebruik
    (ongeacht welke radar er op dat moment actief is, want deze punten zijn radar-onafhankelijk).

    Als de opvraag voor een individueel punt mislukt (MeltingLevelError, bv. netwerkprobleem), wordt
    voor dat punt een resultaat met h0_m/h_minus20_m=None teruggegeven in plaats van de hele functie te
    laten falen - een enkel mislukt punt hoeft niet de classificatie voor de rest van het radarbeeld te
    blokkeren (de betreffende bins worden dan simpelweg als 'geen data' gemarkeerd, zie
    nlr_hclass.classify_hid's NaN-afhandeling).

    Retourneert een lijst met een resultaat-dict per punt in grid_points, in dezelfde volgorde.
    """
    results = []
    for lat, lon in grid_points:
        try:
            r = get_melting_levels(lat, lon, dt_utc, model=model, use_cache=use_cache)
        except MeltingLevelError as e:
            r = {"h0_m": None, "h_minus20_m": None, "source": "error",
                 "model": model, "from_cache": "error: {}".format(e)}
        results.append(r)
    return results


if __name__ == "__main__":
    # Losse test van de interpolatielogica met synthetische data (geen netwerk nodig).
    # Standaard troposferische lapse rate ~6.5 C/km vanaf 15C op zeeniveau:
    # T(h) = 15 - 6.5 * h_km  ->  T = -20 bij h_km = 35/6.5 = 5.3846 km = 5384.6 m
    test_heights = [110, 320, 500, 800, 1000, 1500, 1900, 3000, 4200, 5600, 7200]
    test_temps = [15 - 6.5 * (h / 1000.0) for h in test_heights]
    h20 = _interpolate_minus20_height(test_temps, test_heights)
    expected = 5384.6
    print("Test 1 (lineair standaardprofiel):")
    print("  berekend: {:.1f} m, verwacht: ~{:.1f} m, verschil: {:.1f} m".format(
        h20, expected, abs(h20 - expected)))
    assert abs(h20 - expected) < 5, "interpolatie te onnauwkeurig"

    # Test met ontbrekende (None) waarden ertussen.
    test_temps2 = [15.0, 10.0, None, 0.0, -10.0, None, -25.0]
    test_heights2 = [0, 1000, 1500, 2000, 3000, 3500, 4000]
    h20b = _interpolate_minus20_height(test_temps2, test_heights2)
    print("Test 2 (met ontbrekende waarden): -20C op {:.1f} m".format(h20b))
    assert 3000 < h20b < 4000

    # Test met profiel dat -20C nooit bereikt (te warm/te kort).
    test_temps3 = [15.0, 10.0, 5.0]
    test_heights3 = [0, 500, 1000]
    h20c = _interpolate_minus20_height(test_temps3, test_heights3)
    print("Test 3 (buiten bereik): {}".format(h20c))
    assert h20c is None

    print("\nAlle interpolatietests geslaagd.")
    print("\nLet op: de HTTP-aanroep zelf (get_melting_levels) is hier niet getest -")
    print("open-meteo.com is niet bereikbaar vanuit deze sandbox. Test dat op je eigen machine, bv.:")
    print("  from nlr_meltinglevels import get_melting_levels")
    print("  import datetime")
    print("  r = get_melting_levels(52.36, 6.75, datetime.datetime(2026, 6, 28, 12, 0))")
    print("  print(r)")
