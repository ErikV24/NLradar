# MapTiler live basemap voor NLradar — Fase 1

**Update 8 (opschoning)**: alle diagnostische/probeersel-code uit het "te donker"-traject
verwijderd. Dit is de definitieve, schone staat — geen functionele wijzigingen ten opzichte
van Update 7, alleen opruiming.

## Wat is verwijderd
- De `DEBUG ImageVisual colortransform:`-print uit `vispy/visuals/image.py` — dat bestand
  staat nu weer **exact** in zijn originele staat (er was hier uiteindelijk geen blijvende
  wijziging nodig; de echte fix zit volledig in `nlr_plotting.py`).
- Vier diagnostische prints uit `nlr_maptiles_maptiler.py`'s `get_tiles()` (request bounds,
  gekozen zoom/tegel-range, mozaïek-vorm, reprojectie-uitvoervorm).
- Eén diagnostische print uit `nlr.py`'s `change_basemap_source` (gebruikt om het dubbele-
  aanroep-mysterie op te sporen).

## Wat is behouden (de daadwerkelijke, blijvende fixes uit dit hele traject)
- `nlr_plotting.py`: forceert een colortransform-herziening bij elke tegel-update
  (Update 7's kernfix).
- `nlr_maptiles_maptiler.py`: de aspect-ratio-correctie (Update 3), de kortere/
  foutbestendigere tegel-timeout (Update 5).
- `nlr.py`: de timer-gebaseerde debounce voor het wisselen van kaartbron (Update 5),
  inclusief de hulpfunctie `_apply_basemap_source_change`.
- Alle bestaande foutmeldingen (`tile fetch failed`, `run failed`, etc.) blijven gewoon
  staan — die zijn bedoeld als permanente, nuttige diagnose bij toekomstige problemen, geen
  probeersel.

## MapTiler-stijl: conclusie van het "te donker"-traject
Na uitgebreid onderzoek (zie hierboven, Updates 6-7) bleek de kaart aanvankelijk inderdaad
te donker door een echte bug (colortransform), die nu is opgelost. Een verdere
helderheidsaanpassing via een **eigen, aangepaste MapTiler-kopie** is vervolgens onderzocht,
maar bleek niet haalbaar via de specifieke raster-tegel-API die NLradar gebruikt: dat
eindpunt accepteert alleen MapTiler's eigen, ingebouwde standaardstijlen (zoals
`dataviz-v4-dark`) op het gratis abonnement; eigen kopieën geven een HTTP 403. Dit is een
bevestigde grens van het gratis MapTiler-abonnement, geen NLradar- of code-probleem.
**Huidige, werkende instelling: `dataviz-v4-dark`** (de originele, niet-aangepaste stijl).

---

**Update 7**: waarschijnlijke definitieve oorzaak gevonden van de te-donkere kaart — een
gedrag in vispy zelf dat de verkeerde kleurmethode permanent kan vastzetten voor de hele
sessie. 1 bestand aangepast: `nlr_plotting.py`.

## De cruciale ontdekking
Je laatste console-log liet zien dat mijn diagnostische print (`DEBUG ImageVisual
colortransform:`, toegevoegd in Update 6) **niet** verscheen tijdens jouw hele sessie. Dat
bracht me bij iets fundamenteels in vispy's eigen `ImageVisual`: de keuze tussen "toon dit
gewoon als RGB-kleuren" en "behandel dit als grijswaarde-data en gemiddeld R+G+B/3 voor een
colormap" wordt **slechts één keer** gemaakt — bij de allereerste keer dat er data aan de
kaartlaag wordt gekoppeld, niet steeds opnieuw bij elke `set_data()`-aanroep zoals ik
aannam. Eenmaal gekozen, blijft die keuze **voor de rest van de sessie vastliggen**, ook al
verandert de data daarna volledig.

Dat is belangrijk omdat we al wisten dat jouw lokale tegels niet goed laden (je liet eerder
zien: met "Local" geselecteerd toont NLradar alleen zwart). Bij het opstarten van NLradar
wordt **altijd eerst** de op dat moment ingestelde bron gebruikt voor de allereerste
kaart-data, voordat je ooit de kans krijgt om naar MapTiler te wisselen. Als die allereerste,
mislukte lokale fetch een verkeerd-vormige array teruggeeft, kan de hierboven beschreven
"eenmalige keuze" zich daarop voor de rest van de sessie vastzetten op de grijswaarde-route
— inclusief de `(R+G+B)/3`-gemiddelde-berekening die exact de 1-op-3-verdonkering geeft die
we maten (48 → 16).

## Fix
`Plotting.draw_map_tiles` dwingt nu, bij elke nieuwe tegel-update (dus ook elke keer dat
MapTiler nieuwe tegels aanlevert), een herziening van die kleurmethode-keuze af — in plaats
van te vertrouwen op de (foutieve) aanname dat dit automatisch zou gebeuren. Zo kan een
mogelijk kapotte allereerste lokale fetch niet meer de rest van de sessie blijven
beïnvloeden, voor geen van beide databronnen.

## Eerlijke kanttekening
Dit is de meest overtuigende verklaring die ik tot nu toe heb gevonden, onderbouwd door
vispy's eigen broncode en door uitsluiting van twee eerdere, foutieve theorieën (color
filter, en een verkeerd-toegepaste shader-route binnen dezelfde sessie). Maar omdat ik dit
zelf niet kan draaien, kan ik niet 100% garanderen dat dit het definitieve antwoord is — wel
dat het de sterkste, best onderbouwde kandidaat is die ik kon vinden. De diagnostische print
uit Update 6 staat er nog in: als deze fix werkt, zou je 'm nu juist WEL moeten zien
verschijnen (want we forceren de herziening nu actief), en zou hij 'RGB passthrough path'
moeten melden in plaats van de luminance-route.

---

**Update 6**: een opvallend kleurprobleem (kaart aanzienlijk donkerder dan de bron) gevonden
via exacte pixelmetingen, maar de oorzaak nog niet 100% kunnen bevestigen — een
diagnostische print toegevoegd om dat te doen.

## Wat we hebben vastgesteld
Met exacte pixelmetingen (niet op het oog): een effen wateroppervlak dat in de losse
MapTiler-browserpagina **RGB (48,48,48)** is, kwam in NLradar zelf uit op **RGB (16,16,16)**
— ongeveer een derde van de originele waarde. Dat is te groot om door bestaande, kleine
instellingen (zoals het Color filter, dat met 0.975 nauwelijks effect heeft) verklaard te
worden.

## Sterke aanwijzing, nog niet 100% bevestigd
Diep in vispy's eigen `ImageVisual`-rendercode (`vispy/visuals/image.py`) bestaat een
shader-functie die exact `(R+G+B)/3` berekent — precies de 1-op-3-verhouding die we maten.
Die functie is echter bedoeld voor scalar/grijswaarde-data (zoals de gewone Z/CC/ZDR-
producten), niet voor onze 3-kanaals RGB-kaartdata, die normaliter een andere, neutrale
("passthrough") route zou moeten volgen. Of die twee routes elkaar hier toch op een
onverwachte manier raken, kon ik niet met zekerheid vaststellen zonder zelf te kunnen
draaien (geen PyQt5/vispy-omgeving beschikbaar om dit te testen).

## Toegevoegd: één diagnostische print
Eén regel toegevoegd in `vispy/visuals/image.py` (een bestand dat normaliter nooit wordt
aangepast, want het hoort bij de vispy-bibliotheek zelf, niet bij NLradar's eigen code) die
bij het opbouwen van elke afbeeldingslaag afdrukt welke van de twee routes (RGB-passthrough,
of de grijswaarde/colormap-route met die 1/3-berekening) daadwerkelijk wordt gebruikt, en
met welke datavorm.

**Test opnieuw met MapTiler actief, en stuur de console-output.** Zoek naar een regel die
begint met `DEBUG ImageVisual colortransform:` — die regel zal zeggen of de kaartlaag de
verkeerde (grijswaarde) route volgt. Zo ja, dan hebben we de oorzaak definitief gevonden en
volgt een gerichte fix. Zo nee, dan moeten we verder zoeken, maar dan weten we tenminste
zeker dat dit specifieke spoor niet de oorzaak is.

---

**Update 5**: drie verbeteringen op basis van een uitgebreide console-log analyse — een
kortere/foutbestendigere tegel-timeout, en een fix voor dubbele fetches bij het wisselen
van kaartbron. 2 bestanden aangepast: `nlr_maptiles_maptiler.py`, `nlr.py`.

## 1. Timeout te lang, en blokkeert de hele mozaïek
In de console-log zagen we een paar keer `Read timed out (read timeout=8)`. Bij het oude
gedrag blokkeerde één trage/falende tegel **de hele mozaïek-opbouw** gedurende de volle 8
seconden, en dat kan tijdens snel doorzoomen een paar keer achter elkaar gebeuren — dat
verklaart een deel van de hapering die je tijdens het zoomen zelf voelde.

**Fix**: timeout verkort naar 3 seconden, én een individuele tegel die faalt blokkeert niet
meer de rest — die ene tegel blijft gewoon zwart/leeg, en de overige tegels in de mozaïek
worden nog steeds normaal opgehaald en getoond.

## 2. Dubbele fetch bij het wisselen van Basemap source
In de log zagen we steeds een paar `change_basemap_source called with source= Local` direct
gevolgd door `... source= MapTiler` — dat is geen bug in de keuze zelf (de juiste,
laatste waarde wint altijd), maar wel een **onnodige dubbele** volledige tegel-fetch bij elke
wisseling, vermoedelijk veroorzaakt doordat de Settings-radioknoppen bij het (opnieuw) openen
van het Settings-venster hun begintoestand opnieuw instellen op een manier die het
wijzigingssignaal nog eens afvuurt.

**Fix**: een korte (150ms) "debounce" toegevoegd. Komt er binnen die 150ms een nieuwe
wijziging binnen, dan wordt alleen die laatste daadwerkelijk uitgevoerd — de eerdere,
overbodige aanroep wordt simpelweg geannuleerd voordat hij iets doet.

## Eerlijke kanttekening
Dit pakt twee concrete, in de log waargenomen oorzaken van vertraging/hapering aan, maar ik
kan nog steeds niet garanderen dat dit *alle* hapering wegneemt — vooral de onderliggende
reprojectie-rekentijd zelf (bij elke pan/zoom opnieuw `cv2.remap` over een groot rooster) is
nog niet geoptimaliseerd. Als het na deze update nog steeds duidelijk hapert, vooral tijdens
de beweging zelf, is dat de volgende plek om naar te kijken.

---

**Update 4**: poging om de hapering tijdens pannen/zoomen te verminderen. Alleen
`nlr_plotting.py` aangepast.

## Wat je beschreef
Geen vertraging ná een actie, maar hapering **tijdens** het pannen/zoomen zelf, ook in
gebieden die al in de cache stonden (dus geen netwerk-wachttijd).

## Vermoedelijke oorzaak
De bestaande "wacht tot de gebruiker even stopt"-vertraging (`maptiles_update_time`,
standaard 0.1s) is afgestemd op de lokale tegels, die binnen die 0.1s ruimschoots klaar
zijn. Bij elke kleine muisbeweging tijdens het pannen wordt deze timer **herstart**
(`.stop()` + opnieuw `.start()`), wat bij de lokale tegels geen probleem geeft (zo snel dat
het toch nooit een lopende bewerking onderbreekt), maar bij MapTiler (downloaden +
reprojecteren, ook al komt het grootste deel uit de cache) waarschijnlijk wél: bij een
vloeiende, doorlopende sleepbeweging is 0.1 seconde vaak te kort om de vorige bewerking te
laten afronden voordat de timer alweer wordt herstart — dat zou precies hapering tijdens de
beweging geven, in plaats van een schone update pas na het stoppen.

## Aanpassing
Een langere debounce-tijd (minimaal 0.5s) specifiek wanneer MapTiler de actieve bron is.
Local-tegels blijven op de oorspronkelijke, snelle 0.1s.

## Eerlijke kanttekening
Dit pakt de meest waarschijnlijke oorzaak aan op basis van wat je beschreef, maar ik kan
niet garanderen dat dit de hapering volledig wegneemt — ik kan dit niet zelf live testen.
Als het na deze aanpassing nog steeds hapert, is dat een nuttig signaal dat de oorzaak toch
ergens anders zit (bijvoorbeeld de reprojectie-rekentijd zelf, die dan verder
geoptimaliseerd zou moeten worden, bijvoorbeeld door tijdens het actief pannen/zoomen een
lagere resolutie te gebruiken en die pas na het stoppen te verscherpen).

---

**Update 3**: aspect-ratio (vervormings-)bug gefixt, en de eigen landsgrenzen-/rivierenlaag
wordt nu verborgen wanneer MapTiler actief is. 3 bestanden aangepast: `nlr_maptiles_maptiler.py`,
`nlr_plotting.py`, `nlr.py`.

## Vervormings-bug ("Nederland te steil en te smal")
Gevonden dankzij je beschrijving. De bestaande tekenpijplijn
(`Plotting.get_map_sttransform_parameters`) bepaalt **één enkele** graden-per-pixel-schaal
op basis van de hoogte van de kaartafbeelding, en past die schaal **ook toe op de breedte**.
Dat klopt automatisch bij de lokale tegels (die zijn opgebouwd uit gelijke vierkante
tegels), maar mijn MapTiler-module stelde breedte en hoogte onafhankelijk van elkaar in
(gewoon de schermgrootte) — zonder te garanderen dat die verhouding ook echt overeenkwam
met de gevraagde lat/lon-breedte/hoogte. Dat gaf precies dit soort "te steil/te smal"-
vervorming.

**Fix**: de uitvoerresolutie wordt nu zo berekend dat de graden-per-pixel-schaal op beide
assen exact gelijk is — er wordt gekeken welke as (breedte- of hoogte-gestuurd) de scherpste
resulterende afbeelding geeft, en die wordt gebruikt, maar altijd met een correcte
verhouding.

## Eigen grenzenlaag verborgen bij MapTiler
Op jouw verzoek: de landsgrenzen/provincies/rivieren die NLradar zelf tekent
(`map_lines`-laag) worden nu volledig verborgen zodra Basemap source op MapTiler staat,
omdat die kaart zijn eigen grenzen/labels al toont en dubbele lijnen alleen maar verwarrend
zouden zijn. Je bestaande instellingen voor welke lijnen normaal getoond worden (Countries/
Provinces/Rivers-vakjes) blijven onaangeroerd — schakel je terug naar "Local", dan komen ze
gewoon weer terug zoals ze stonden.

---

**Update 2**: de daadwerkelijke oorzaak van de verschoven/vervormde kaart gevonden — een
**dubbele projectie**. Alleen `nlr_maptiles_maptiler.py` opnieuw aangepast (en flink
vereenvoudigd).

## De echte oorzaak
Dankzij de console-output (download en reprojectie liepen technisch foutloos door) kon ik
uitsluiten dat het downloaden zelf het probleem was. Het echte probleem: ik liet mijn module
zelf de Mercator → AEQD-projectie uitvoeren, **terwijl NLradar's bestaande tekensysteem dat
óók al doet** — automatisch, op de GPU, voor elke kaartlaag (`Plotting.draw_map_tiles` +
`self.map_transforms['aeqd']`). Dat systeem verwacht dat `self.map_data` een simpel,
recht lat/lon-rooster is (precies zoals de bestaande lokale tegels al zijn opgebouwd) —
niet een al-geprojecteerde AEQD-afbeelding.

Door zelf ook al naar AEQD te projecteren, werd de data **twee keer** geprojecteerd: eenmaal
door mij (CPU), eenmaal door het bestaande systeem (GPU). Die dubbele, niet-overeenkomende
projectie is precies wat de scheve/verschoven plaatsnamen en de algehele vervorming
verklaarde die je op het screenshot zag.

## Fix
`_reproject_to_aeqd` (Mercator → AEQD) is vervangen door `_reproject_to_latlon_grid`
(Mercator → simpel, recht lat/lon-rooster — fysiek een veel eenvoudigere bewerking, en
nu ook minder code). De AEQD-projectie gebeurt voortaan weer precies één keer, door het
bestaande, al jarenlang werkende systeem — exact zoals het ook voor de lokale tegels werkt.

Bijkomend voordeel: de `nlr_functions`-afhankelijkheid (en de AEQD-rekenwerk daarin) is niet
meer nodig in dit bestand, dus dat is ook verwijderd — minder code, minder kans op een
volgende dit-soort-fout.

## Gevalideerd
Met een synthetisch testpatroon (gekleurd rooster + rode/gele referentie-breedte-/
lengtegraadlijnen), met **exact dezelfde grenzen en schermgrootte als in jouw eigen
console-log** (zoom 7, tegel-range 64-67/40-43, 710×564px) — de referentielijnen komen er nu
**recht** uit (zoals een zuiver lat/lon-rooster moet zijn), in plaats van de kromming die we
bewust testten (en wilden zien) in de vorige, onjuiste versie.

---

**Update**: vermoedelijke oorzaak gevonden waarom er bij de eerste test (geen foutmeldingen,
geen wisseling) niets gebeurde, plus diagnostische prints toegevoegd om het zeker te weten.

## Gevonden probleem: radioknoppen-volgorde
De twee nieuwe radioknoppen ("Local"/"MapTiler") riepen hun `change_basemap_source(...)`-
functie aan zonder te kijken naar de daadwerkelijke aan/uit-status van de knop (een
`lambda: ...` die de `toggled`-parameter negeerde). Bij een `QButtonGroup` vuurt het
`toggled`-signaal voor **beide** knoppen bij elke wisseling — als de volgorde waarin Qt dat
doet net andersom was dan ik aannam, won de oude instelling alsnog, en bleef de hele
MapTiler-code onbereikt (geen download, geen foutmelding, want de functie werd letterlijk
niet aangeroepen met de juiste waarde).

**Fix**: de handlers gebruiken nu de daadwerkelijke `checked`-boolean uit het signaal, en
doen alleen iets als de knop *aan* gaat — niet meer afhankelijk van signaalvolgorde.

## Diagnostische prints toegevoegd
Om dit zeker te bevestigen (of uit te sluiten als het na de fix nog steeds niet werkt) staan
er nu prints bij:
- `nlr.py`'s `change_basemap_source` — bevestigt of de instelling-wijziging zelf wordt
  opgepikt.
- `nlr_maptiles_maptiler.py`'s `get_tiles` — bevestigt of de tegel-aanvraag wordt gestart,
  welke zoom/tegel-range gekozen wordt, en of de download + reprojectie voltooien.

Test opnieuw, en stuur de console-output — met deze prints zouden we nu exact moeten zien
tot waar het komt, ook als er nog een vervolgprobleem is.

---

Een nieuwe, optionele live/meescrollende kaartlaag (zoals Bram gebruikt), naast de bestaande
lokale, statische tegels — instelbaar via Settings → Map → "Basemap source".

**4 bestanden, waarvan 1 nieuw:**
- `nlr_maptiles_maptiler.py` — **nieuw bestand**, de kernlogica (tegels downloaden + naar de
  juiste projectie omrekenen)
- `nlr.py` — Settings-GUI (kaartbron-keuze, MapTiler-stijl-ID, API-key-veld)
- `nlr_globalvars.py` — MapTiler geregistreerd als nieuwe API-key-databron
- `nlr_plotting.py` — koppeling tussen de bestaande teken-logica en de nieuwe tegelbron

## Belangrijk: wat ik wél en niet heb kunnen testen

Mijn omgeving heeft geen toegang tot `api.maptiler.com` (netwerk-egress is geblokkeerd) en
geen PyQt5/vispy om de GUI te draaien. Dat betekent:

**Wél getest (met standalone Python-scripts, los van de PyQt5/vispy-app):**
- De Web Mercator tegel-coördinatenberekening (`latlon_to_mercator_tile_xy`) — kwam exact
  overeen met een onafhankelijke handmatige berekening.
- De zoom-niveau-selectie (`choose_zoom_level`) — gedraagt zich logisch voor een aantal
  test-scenario's.
- **De volledige Mercator → AEQD reprojectie-pijplijn**, met een synthetisch testpatroon
  (een gekleurd tegelrooster + referentie-lat/lon-lijnen). Op een groot gebied (500+ km)
  was duidelijk de verwachte vervorming/kromming van het rooster te zien na reprojectie —
  het bewijs dat de projectiecorrectie daadwerkelijk iets doet, in de juiste richting.
- Tijdens dit testen vond ik een echte bug (shape-mismatch bij scalar/array-mengvormen in
  `latlon_to_mercator_tile_xy`) en heb die gefixt.

**NIET getest (kon ik niet, vanwege de netwerk-/PyQt5-beperking):**
- De daadwerkelijke download van een MapTiler-tegel (de URL/key-aanroep zelf).
- Of de gedownloade tegel-mozaïek correct in elkaar wordt gezet (`_fetch_mercator_mosaic`).
- De volledige integratie in de Settings-GUI (of de radioknoppen/velden zich gedragen zoals
  bedoeld, of de juiste signalen afgaan).
- Of de uiteindelijke kaart er op het scherm goed uitziet binnen de echte NLradar-applicatie.

Dit is dus typisch een feature waarbij ik jou nodig heb om de laatste, cruciale stap te
zetten: gewoon proberen, en de console-output + een screenshot delen zoals we ook bij de
PolRGB-feature deden.

## Hoe te gebruiken

1. **MapTiler API-key invullen**: Settings → Download → API keys → MapTiler. Plak je key
   (het stuk na `key=` uit de URL die je me liet zien, dus `R696KEq79OsQA2EOiRy7...`).
2. **Stijl controleren**: Settings → Map → "MapTiler map style/ID" staat standaard op
   `dataviz-v4-dark`. Pas dit aan als je een andere MapTiler-kaart wil gebruiken (de waarde
   is het deel van de URL tussen `/maps/` en `/style.json`).
3. **Schakelen**: Settings → Map → "Basemap source" → kies "MapTiler" in plaats van "Local".
4. De kaart zou na een paar seconden (download + reprojectie) moeten verschijnen.

## Wat er technisch gebeurt (samengevat)

MapTiler levert kaarttegels in de **Web Mercator**-projectie (de gebruikelijke "platte"
kaartprojectie die vrijwel elke webkaart gebruikt). NLradar's bestaande kaartweergave
gebruikt een **Azimuthal Equidistant**-projectie gecentreerd op de actieve radar (nodig om
afstanden vanaf de radar correct weer te geven). Dit zijn fundamenteel verschillende
projecties — een MapTiler-tegel zomaar uitrekken zou een toenemende vervorming geven
naarmate je verder van de radar kijkt.

`nlr_maptiles_maptiler.py` voorkomt dat door, voor elke pixel van de uiteindelijke kaart,
terug te rekenen welke lat/lon daar moet komen, die lat/lon op te zoeken in de gedownloade
Mercator-tegels, en dat met `cv2.remap` (snel, vectorized) samen te voegen tot het
eindbeeld — in plaats van de tegels simpelweg te showen zoals ze zijn.

## Caching

Gedownloade tegels worden lokaal opgeslagen in `Generated_files/maptiler_tile_cache/`, zodat
je niet steeds opnieuw dezelfde tegel van internet hoeft te halen. Als je een tegel ooit
corrupt/verkeerd binnenkrijgt, verwijder dan gewoon dat bestand (of de hele map) — het wordt
automatisch opnieuw gedownload.

## Bekende beperkingen / nog open voor latere fases

- **Geen automatische fallback**: als de MapTiler-download faalt (geen internet, verlopen
  key, etc.), blijft het kaartpaneel leeg/zwart in plaats van automatisch terug te schakelen
  naar de lokale tegels. Foutmeldingen verschijnen wel in de console.
- **Performance niet geoptimaliseerd**: bij elke pan/zoom wordt de volledige reprojectie
  opnieuw berekend (geen gedeeltelijke updates/caching van het gereprojecteerde resultaat
  zelf, alleen van de losse Mercator-tegels). Op een grote/langzame machine kan dit
  merkbaar trager aanvoelen dan de lokale tegels.
- **`MAX_TILES_PER_AXIS = 12`**: een ingebouwde limiet om te voorkomen dat een bug elders
  honderden tegel-downloads tegelijk veroorzaakt. Bij een zeer ingezoomde weergave op een
  hoog zoomniveau zou dit een merkbaar lagere resolutie kunnen geven dan verwacht — meld het
  als dat opvalt, dan stemmen we de limiet of de zoom-keuze verder af.
- **Geen automatische verwijdering van oude cache-tegels**: de cachemap groeit ongelimiteerd.
  Voor nu handmatig leegmaken indien nodig; een automatisch opruimmechanisme is een mogelijke
  toevoeging voor een latere fase.

## Volgende stappen (afhankelijk van hoe deze eerste test verloopt)

- Als de tegel-download zelf niet werkt: stuur de exacte foutmelding (die zou nu, dankzij de
  uitgebreide foutafhandeling in `_fetch_tile`, een duidelijke reden moeten geven — verkeerde
  key, verkeerde stijl-ID, of een andere HTTP-fout).
- Als de download werkt maar de kaart er verkeerd uitziet (verschoven, vervormd, verkeerde
  kleuren): een screenshot zegt meer dan een beschrijving, zoals we wel vaker hebben gedaan.
- Als alles werkt: dan kunnen we in een volgende fase kijken naar performance-optimalisatie,
  een nettere fallback bij fouten, en eventueel cache-opruiming.
