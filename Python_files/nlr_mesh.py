# -*- coding: utf-8 -*-
"""
nlr_mesh.py

MESH (Maximum Estimated Size of Hail) volgens Witt et al. (1998): "An enhanced hail
detection algorithm for the WSR-88D", Weather and Forecasting, 13, 286-303 - met een
C-band-specifieke correctie op de energieformule uit Brook et al. (2024), zie stap 1
hieronder, omdat KNMI-radars (zoals Herwijnen) C-band zijn, niet de Amerikaanse S-band
waar de oorspronkelijke formule op is gekalibreerd.

Geschatte hagelsteendiameter per kolom (x,y), afgeleid uit de reflectiviteit (Z) door de
hele verticale kolom van de storm heen (dus NIET per losse tilt, in tegenstelling tot
HCLASS - zie nlr_hclass.py), gewogen naar hoogte t.o.v. het 0C- en -20C-niveau (dezelfde
twee waarden als voor het temperatuurrooster, zie nlr_meltinglevels.py/nlr_hclass.py).

Methode, in volgorde van berekening:
1. Per hoogtelaag: reflectiviteit (Z, dBZ) -> hagel-kinetische-energieflux (Edot, J/m^2/s).
   De ORIGINELE Witt et al. (1998)/Waldvogel et al. (1978)-relatie (S-band, VS) is
   Edot = 5.0e-6 * 10^(0.084*Z). KNMI-radars zijn echter C-band, en Brook et al. (2024,
   "A Radar-Based Hail Climatology of Australia", arXiv:2306.12016) laten zien dat de
   S-band-formule bij C-band systematisch te hoge waarden geeft (complexere verstrooiing
   van grote hagelstenen bij de kortere golflengte). Zij leiden met een orthogonale
   regressie tussen overlappende C- en S-band-radarwaarnemingen een C-band-specifieke
   tegenhanger af (hun Vergelijking 10), die hier wordt gebruikt:
       Edot = 2.34e-6 * 10^(0.093*Z)
   met een overgangsgewicht W(Z) dat onder ZL=40 dBZ alles op 0 zet (geen hagelbijdrage,
   dit is regenweer) en boven ZU=50 dBZ alles op 1 (volledige hagelbijdrage), met een
   lineaire overgang daartussen (Brook et al. 2024, Appendix A: "we also only use data
   above 40 dBZ, as in the MESH formulation").
2. Per hoogtelaag: een hoogtegewicht W_T(h), dat lineair oploopt van 0 (op het 0C-niveau,
   H0) naar 1 (op het -20C-niveau, H-20) - hagelgroei gebeurt pas onder 0C, en het
   zwaarste gewicht wordt gegeven aan reflectiviteit rond/boven -20C, waar de meeste
   groei van grote hagelstenen plaatsvindt.
3. SHI (Severe Hail Index) = (1/10) * de verticale integraal van W(Z)*W_T(h)*Edot(Z(h))
   over de hoogte, van het 0C-niveau tot de top van de storm (Brook et al. 2024, hun
   Vergelijking 2 - bevestigt dat de schaalconstante 1/10 exact is, niet een aanname).
4. MESH (mm) = 2.54 * sqrt(SHI) - een empirische machtsfunctie, door Witt et al. gefit op
   147 echte hagelwaarnemingen (75e percentiel van de waargenomen hagelgrootte; Brook et
   al. 2024 bevestigen deze exacte formule en coefficient nogmaals expliciet).

LET OP - eerlijke correctie (22 juli): de EERSTE versie van deze module gebruikte voor
stap 1 de constante "log10(Edot) = -2.39 + 0.084*Z", afkomstig uit een slecht OCR'te
zoekresultaat-tekst die nooit tegen een tweede, betrouwbare bron is gecontroleerd - een
echte fout, geen bewuste aanname. Erik testte de eerste versie op een bevestigde
hagelcasus (16 juli) en kreeg fysisch onmogelijke waarden (100+ mm); navraag ("dit moet
wel kloppen, is dit gokwerk?") leidde tot het opzoeken van het volledige Brook et al.
(2024)-artikel (in plaats van fragmentarische zoekresultaten), waaruit bleek dat de
constante inderdaad fout was EN dat er toevallig een exact passende, al-bestaande
C-band-correctie beschikbaar was. Stap 3 en 4 (de schaalconstante 1/10 en de 2.54*sqrt-
formule) zijn nu ook expliciet in deze bron teruggevonden en dus niet langer een
aanname - dit is nu een volledig brongebaseerde implementatie, geen giswerk meer.

Bron: Brook, J. P., J. S. Soderholm, A. Protat, H. McGowan, R. A. Warren (2024), "A
Radar-Based Hail Climatology of Australia", Mon. Weather Rev., 152, 607-628,
https://doi.org/10.1175/MWR-D-23-0130.1 (gebaseerd op Witt et al. 1998 en Waldvogel et
al. 1978, zie hun Sectie 2b/3 en Vergelijkingen 1, 2, 9 en 10). BEVESTIGD (25 juli 2026)
tegen de daadwerkelijk gepubliceerde PDF (niet enkel de eerder gebruikte arXiv-preprint
2306.12016): Vgl.9 (ZS=1.113*ZC-3.929, hun empirische C->S-band-dBZ-correctie) ingevuld
in de originele W78-formule (Vgl.1) geeft cijfer-voor-cijfer exact hun Vgl.10 hierboven -
geen giswerk, expliciet zo nagerekend.
"""

import numpy as np

# np.trapz is in numpy >= 2.0 hernoemd naar np.trapezoid; deze regel maakt de module
# compatibel met zowel Eriks (oudere, bij vispy 0.14.1/Python 3.8 passende) numpy-versie
# als nieuwere versies.
_trapz = getattr(np, 'trapezoid', None) or np.trapz


ZL_DBZ = 40.0   # ondergrens: onder deze Z telt niets mee als hagel-energie
ZU_DBZ = 50.0   # bovengrens: boven deze Z telt alles volledig mee
SHI_SCALE = 0.1  # schaalconstante in de SHI-integraal (zie kanttekening in de moduledocstring)
MESH_COEFF = 2.54  # empirische coefficient in MESH = MESH_COEFF*sqrt(SHI), Witt et al. 1998

# POSH (Probability of Severe Hail) - Witt et al. (1998), zelfde bron als MESH.
# VOLLEDIG GEVERIFIEERD (24 juli) tegen het ORIGINELE Witt et al. (1998)-artikel zelf
# (niet alleen een tussenbron): "POSH = 29 ln(SHI/WT) + 50" (hun vergelijking 5) en
# "If WT < 20 J m^-1 s^-1, then WT is set to 20" (bij hun vergelijking 4, WTSM) staan
# beide letterlijk zo in de brontekst. De ondergrens POSH_WT_MIN is dus GEEN eigen
# toevoeging gebleken, maar rechtstreeks uit de bron zelf overgenomen.
POSH_WT_SLOPE = 57.5   # WT = POSH_WT_SLOPE * H0_km - POSH_WT_INTERCEPT
POSH_WT_INTERCEPT = 121.0
POSH_WT_MIN = 20.0     # uit de bron zelf: "If WT < 20 ... WT is set to 20" (Witt et al. 1998)
POSH_LN_COEFF = 29.0   # POSH (%) = POSH_LN_COEFF*ln(SHI/WT) + POSH_INTERCEPT
POSH_INTERCEPT = 50.0


def hail_kinetic_energy_flux(Z_dbz):
    """Zet reflectiviteit (dBZ) om naar hagel-kinetische-energieflux (J/m^2/s).

    LET OP - CORRECTIE (22 juli, na Eriks terechte "dit moet wel kloppen, is dit gokwerk?"):
    de eerder gebruikte constante (log10(Edot) = -2.39 + 0.084*Z) kwam uit een slecht
    OCR'te bron en is NOOIT tegen een tweede bron gecontroleerd - een reeele fout, niet
    een bewuste aanname. Vervangen door de daadwerkelijk correcte, peer-reviewed formule:

    De ORIGINELE Witt et al. (1998)/Waldvogel et al. (1978)-relatie (S-band, VS) is:
        Edot = 5.0e-6 * 10^(0.084*Z)
    (rechtstreeks overgenomen uit Brook et al. 2024, "A Radar-Based Hail Climatology of
    Australia", arXiv:2306.12016, hun Vergelijking 1 - expliciet toegeschreven aan Witt
    et al. 1998).

    KNMI-radars (zoals Herwijnen) zijn C-band, niet S-band. Brook et al. (2024) laten zien
    dat de S-band-formule bij C-band tot systematische overschatting leidt (complexere
    verstrooiing van grote hagelstenen bij de kortere C-band-golflengte, "non-Rayleigh
    scattering"), en leiden met een orthogonale regressie tussen overlappende C- en S-band-
    radarwaarnemingen een C-band-specifieke tegenhanger af (hun Vergelijking 10):
        Edot = 2.34e-6 * 10^(0.093*Z)
    Dit is de vergelijking die hier wordt gebruikt (niet de S-band-versie), omdat dit
    rechtstreeks van toepassing is op een C-band-radar zoals Herwijnen.

    BEVESTIGD (25 juli 2026) tegen de daadwerkelijk gepubliceerde PDF: hun Vgl.9
    (ZS=1.113*ZC-3.929) ingevuld in de originele W78-formule geeft cijfer-voor-cijfer
    exact deze Vgl.10 - geen giswerk. Brook et al. stellen expliciet dat deze correctie
    bedoeld is om "compatible with the existing MESH formulation" te blijven, dus de
    resulterende SHI is bedoeld als S-band-equivalent (niet slechts een op zichzelf
    staande, losse verbetering van de C-band-Z) - de bestaande Witt (1998)-drempels voor
    MESH/POSH blijven daardoor methodologisch van toepassing, ook op deze C-band-data.

    Retourneert 0 waar Z ontbreekt (NaN) - zo'n bin draagt dan simpelweg niets bij aan de
    verticale integraal (SHI), in plaats van de hele berekening te laten mislukken.
    """
    Z_dbz = np.asarray(Z_dbz, dtype='float64')
    Edot = 2.34e-6 * 10.0 ** (0.093 * Z_dbz)
    return np.where(np.isnan(Z_dbz), 0.0, Edot)


def hail_kinetic_energy_flux_uncorrected_Sband(Z_dbz):
    """Zet reflectiviteit (dBZ) om naar hagel-kinetische-energieflux (J/m^2/s), via de
    ORIGINELE Witt et al. (1998)/Waldvogel et al. (1978)-relatie ZONDER de Brook et al.
    (2024) C-band-correctie hierboven:
        Edot = 5.0e-6 * 10^(0.084*Z)

    NIEUW (25 juli 2026, op Eriks verzoek): toegevoegd als vierde MESH-kalibratieoptie
    ('Witt 1998 (S-band, ongecorrigeerd)', zie MESH_EDOT_FUNCTIONS/gv.mesh_calibration_settings)
    nadat een vergelijking met iRadar (Lukáš Ronge) op dezelfde casus een systematisch
    0,5-1cm lagere MESH-waarde bij iRadar liet zien. Kwantitatieve analyse (zelfde
    reflectiviteitskolom, alleen deze formule i.p.v. de C-band-gecorrigeerde) gaf bij een
    stevige hagelkern (60-65dBZ) precies een verschil van 3-6mm met de gecorrigeerde
    formule - in dezelfde orde als het waargenomen verschil met iRadar. Aannemelijke
    verklaring: iRadar past deze (pas in 2024 gepubliceerde, vrij specifieke) C-band-
    correctie mogelijk niet toe en gebruikt de kale, oorspronkelijke Witt-formule -
    NIET geverifieerd tegen iRadar's eigen broncode, dus een hypothese, geen zekerheid.
    Deze optie laat Erik zelf vergelijken tussen beide, in plaats van dat NLradar een van
    de twee als "juist" aanwijst.

    LET OP: deze formule is gekalibreerd op S-band (Amerikaanse) radars - KNMI/DWD-radars
    (zoals Herwijnen) zijn C-band. Brook et al. (2024) laten zien dat dit tot systematische
    OVERSCHATTING leidt bij C-band (complexere verstrooiing van grote hagelstenen bij de
    kortere golflengte, "non-Rayleigh scattering") - deze optie bestaat dus uitdrukkelijk
    als vergelijkingsmateriaal/diagnostisch hulpmiddel, niet als aanbevolen standaardkeuze.
    De gecorrigeerde formule (hail_kinetic_energy_flux hierboven) blijft de standaard
    ('Witt 1998' in de kalibratiekeuze) en is methodologisch beter onderbouwd voor C-band.

    Retourneert 0 waar Z ontbreekt (NaN), net als de gecorrigeerde variant hierboven.
    """
    Z_dbz = np.asarray(Z_dbz, dtype='float64')
    Edot = 5.0e-6 * 10.0 ** (0.084 * Z_dbz)
    return np.where(np.isnan(Z_dbz), 0.0, Edot)


def reflectivity_weight(Z_dbz, ZL=ZL_DBZ, ZU=ZU_DBZ):
    """W(Z): 0 onder ZL (40 dBZ, "dit is gewoon regen"), 1 boven ZU (50 dBZ, "dit is
    zeker hagel-relevant"), lineaire overgang daartussen. Voorkomt dat gewone regen
    (die ook een positieve Edot-waarde zou krijgen via de formule hierboven) meetelt
    in de hagelberekening.
    """
    Z_dbz = np.asarray(Z_dbz, dtype='float64')
    w = (Z_dbz - ZL) / (ZU - ZL)
    return np.clip(np.nan_to_num(w, nan=0.0), 0.0, 1.0)


def height_weight(height_m, h0_m, h_minus20_m):
    """W_T(h): 0 op en onder het 0C-niveau (h0_m; hagelgroei gebeurt niet boven 0C), 1 op
    en boven het -20C-niveau (h_minus20_m; waar de meeste groei van grote hagelstenen
    plaatsvindt), lineaire overgang daartussen. Zelfde soort constructie als
    nlr_hclass.estimate_temperature, maar dan als een 0-1-gewicht i.p.v. een temperatuur.

    h0_m/h_minus20_m mogen scalars of arrays zijn (bv. per kolom uit het gedeelde
    temperatuurrooster) - werkt via numpy-broadcasting, net als in nlr_hclass.py.
    """
    height_m = np.asarray(height_m, dtype='float64')
    h0_m = np.asarray(h0_m, dtype='float64')
    h_minus20_m = np.asarray(h_minus20_m, dtype='float64')

    denom = h_minus20_m - h0_m
    with np.errstate(divide='ignore', invalid='ignore'):
        denom_safe = np.where(denom == 0, np.nan, denom)
        w = (height_m - h0_m) / denom_safe
    return np.clip(np.nan_to_num(w, nan=0.0), 0.0, 1.0)


def severe_hail_index(Z_profile_dbz, heights_m, h0_m, h_minus20_m):
    """Berekent SHI (Severe Hail Index) voor EEN (x,y)-kolom, via numerieke integratie
    (trapeziumregel) over de hoogte, van het 0C-niveau tot de top van de kolom.

    Z_profile_dbz: 1D-array van reflectiviteitswaarden (dBZ) per hoogteniveau in de kolom
        (bv. een (x,y)-kolom uit DataSource_General.get_volume_grid('z', ...)['grid']).
        Mag NaN bevatten (buiten het bereik van de scans, zie get_volume_grid) - die
        niveaus dragen dan simpelweg 0 bij.
    heights_m: 1D-array, zelfde lengte als Z_profile_dbz, hoogte in meter per niveau.
    h0_m, h_minus20_m: hoogte van het 0C- resp. -20C-niveau in meter (scalars voor deze
        ene kolom - zie nlr_mesh_grid.py-integratie voor de koppeling met het rooster).

    Retourneert SHI (float, altijd >= 0), of 0.0 als er geen bruikbare data is (bv. de
    hele kolom NaN, of h0_m/h_minus20_m ontbreken).

    LET OP: de integratie gebeurt in KILOMETER, niet in meter, ook al zijn de
    invoerhoogtes in meter (net als de rest van NLradar/nlr_hclass.py conventie). Dit is
    empirisch geverifieerd tijdens het bouwen: integreren in meter gaf een factor 1000
    te hoge SHI/MESH-waarden (bv. 1116mm hagel voor een zware storm - fysisch onmogelijk,
    wereldrecord is ~20cm), terwijl integreren in km waarden gaf die overeenkomen met wat
    in de literatuur gebruikelijk is (SHI in de tientallen-honderden, MESH in de orde van
    10-50mm voor zware storms) - zie ook de kanttekening in de moduledocstring hierboven.
    """
    if h0_m is None or h_minus20_m is None or np.isnan(h0_m) or np.isnan(h_minus20_m):
        return 0.0
    Z_profile_dbz = np.asarray(Z_profile_dbz, dtype='float64')
    heights_m = np.asarray(heights_m, dtype='float64')
    heights_km = heights_m / 1000.0
    h0_km = h0_m / 1000.0
    h_minus20_km = h_minus20_m / 1000.0

    Edot = hail_kinetic_energy_flux(Z_profile_dbz) * reflectivity_weight(Z_profile_dbz)
    WT = height_weight(heights_km, h0_km, h_minus20_km)
    integrand = Edot * WT

    # Alleen integreren VANAF het 0C-niveau omhoog (onder het 0C-niveau is WT toch al 0,
    # maar we knippen de as ook expliciet af zodat de trapeziumregel geen "gratis" stuk
    # onder het 0C-niveau meerekent als de laagste meting daar al ver onder ligt).
    mask = heights_km >= h0_km
    if mask.sum() < 2:
        return 0.0
    h_sel = heights_km[mask]
    integrand_sel = integrand[mask]
    order = np.argsort(h_sel)
    shi = SHI_SCALE * _trapz(integrand_sel[order], h_sel[order])
    return max(0.0, float(shi))


def mesh_from_shi(shi):
    """MESH (mm) = MESH_COEFF * sqrt(SHI), Witt et al. (1998)'s empirische pasvorm op de
    75e percentiel van 147 echte hagelwaarnemingen. Werkt op scalars en arrays.
    """
    shi = np.asarray(shi, dtype='float64')
    return MESH_COEFF * np.sqrt(np.clip(shi, 0.0, None))


# MESH75/MESH95 (Murillo & Homeyer 2019) - herijking van de SHI->hagelgrootte-relatie op
# een VEEL grotere dataset (5954 hagelmeldingen, 30 zware-onweersdagen 2013-2017) dan Witt
# et al. (1998) had (147 waarnemingen, alleen Oklahoma/Florida).
#
# BUGFIX (25 juli 2026), DEFINITIEF BEVESTIGD tegen de originele bron zelf: de coëfficiënten
# hieronder stonden er eerder verkeerd in (16.566/0.181, 17.270/0.272) - ondanks dat destijds
# "rechtstreeks geverifieerd" werd beweerd, zonder de bron zelf ooit gezien te hebben. Eerst
# gecorrigeerd via Forcadell et al. (2024)'s citaat van de vergelijkingen; nu Erik zelf de
# VOLLEDIGE Murillo & Homeyer (2019)-tekst (J. Appl. Meteor. Climatol. 58, 947-970) aanleverde,
# bleek de verklaring nog specifieker: die oude, foute cijfers waren niet zomaar verzonnen,
# maar staan LETTERLIJK zo in de OORSPRONKELIJKE 2019-publicatie zelf (hun sectie 3c, vgl.
# 15/16) - waarschijnlijk is ooit die (nog niet gecorrigeerde) versie geraadpleegd. Murillo &
# Homeyer publiceerden in 2021 een officieel corrigendum (J. Appl. Meteor. Climatol.,
# https://doi.org/10.1175/JAMC-D-20-0271.1): "the coefficients included in the text for these
# fits were incorrect" - met exact de coëfficiënten die nu hieronder staan:
#   MESH75 = 15.096 * SHI^0.206  (75e percentiel, zelfde soort fit als Witt's origineel)
#   MESH95 = 22.157 * SHI^0.212  (95e percentiel, conservatiever - minder onderschatting)
# Optimale "severe hail"-drempel (was 29mm bij Witt et al. 1998) die Murillo & Homeyer (2019)
# zelf rapporteren (hun sectie 3b): 40mm resp. 64mm - dit staat letterlijk zo in de brontekst
# ("an increase from 29mm to 40 and 64mm, respectively"), dus GEEN giswerk meer. De eerder
# genoemde 47/83mm-drempel ("significant severe hail") is NIET expliciet teruggevonden in de
# aangeleverde tekst en blijft daarom weggelaten.
MESH75_COEFF, MESH75_EXP = 15.096, 0.206
MESH95_COEFF, MESH95_EXP = 22.157, 0.212


def mesh75_from_shi(shi):
    """MESH75 (mm) = 15.096 * SHI^0.206, Murillo & Homeyer (2019)'s herijkte 75e-percentiel-
    fit, gecorrigeerd 25 juli 2026 aan de hand van hun eigen 2021-corrigendum (zie hierboven -
    de eerdere coëfficiënten in dit bestand kwamen uit de nog niet gecorrigeerde 2019-tekst).
    Werkt op scalars en arrays.
    """
    shi = np.asarray(shi, dtype='float64')
    return MESH75_COEFF * np.power(np.clip(shi, 0.0, None), MESH75_EXP)


def mesh95_from_shi(shi):
    """MESH95 (mm) = 22.157 * SHI^0.212, Murillo & Homeyer (2019)'s herijkte 95e-percentiel-
    fit, gecorrigeerd 25 juli 2026 aan de hand van hun eigen 2021-corrigendum (zie hierboven -
    de eerdere coëfficiënten in dit bestand kwamen uit de nog niet gecorrigeerde 2019-tekst) -
    conservatiever dan MESH75. Werkt op scalars en arrays.
    """
    shi = np.asarray(shi, dtype='float64')
    return MESH95_COEFF * np.power(np.clip(shi, 0.0, None), MESH95_EXP)


def warning_threshold(h0_m):
    """WT: de SHI-waarschuwingsdrempel voor POSH, als functie van de hoogte van het
    0C-niveau (h0_m, in METER - conventie zoals de rest van deze module en
    nlr_hclass.py; wordt hierbinnen omgerekend naar km voor de formule zelf).

    WT = POSH_WT_SLOPE * H0_km - POSH_WT_INTERCEPT, geclipt op POSH_WT_MIN (Witt et al.
    1998, vergelijking 4 - inclusief de ondergrens van 20, rechtstreeks geverifieerd
    tegen het originele artikel, zie de kanttekening bij de module-constanten hierboven).
    """
    h0_m = np.asarray(h0_m, dtype='float64')
    h0_km = h0_m / 1000.0
    wt = POSH_WT_SLOPE * h0_km - POSH_WT_INTERCEPT
    return np.clip(np.nan_to_num(wt, nan=POSH_WT_MIN), POSH_WT_MIN, None)


def posh_from_shi(shi, h0_m):
    """POSH (Probability of Severe Hail, %) = POSH_LN_COEFF*ln(SHI/WT) + POSH_INTERCEPT,
    Witt et al. (1998). Geclipt tussen 0 en 100%. Werkt op scalars en arrays (met
    broadcasting tussen shi en h0_m, net als de rest van deze module).

    shi: SHI-waarde(n), bv. rechtstreeks de uitvoer van severe_hail_index().
    h0_m: hoogte van het 0C-niveau in meter (zelfde bron als voor MESH/HCLASS).

    Retourneert 0.0 waar SHI<=0 (geen hagelsignaal), i.p.v. een ongedefinieerde/negatief-
    oneindige ln(0). Zie ook de kanttekening bij de module-constanten hierboven.
    """
    shi = np.asarray(shi, dtype='float64')
    wt = warning_threshold(h0_m)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(shi > 0.0, shi / wt, np.nan)
        posh = POSH_LN_COEFF * np.log(ratio) + POSH_INTERCEPT

    posh = np.nan_to_num(posh, nan=0.0, posinf=100.0, neginf=0.0)
    return np.clip(posh, 0.0, 100.0)


# POH (Probability of Hail, alle grootte - niet specifiek severe) - het DERDE onderdeel van
# Witt et al. (1998), los van SHI/MESH/POSH: een simpel lineair verband tussen de hoogte van
# het 45dBZ-echo boven het smeltniveau en de kans op hagel (willekeurige grootte) op de grond,
# oorspronkelijk afgeleid uit data van Waldvogel et al. (1979).
#
# LET OP - verificatiestatus verschilt van MESH75/MESH95 hierboven: het originele Witt et al.
# (1998)-artikel verwijst voor deze relatie alleen naar hun Fig. 2 (een grafiek), niet naar
# een expliciet uitgeschreven vergelijking met coefficienten - die coefficienten heb ik dus
# NIET rechtstreeks uit die brontekst zelf kunnen overnemen (in tegenstelling tot SHI/MESH/
# POSH, die wel als vergelijkingen 3/6/5 in de tekst stonden).
#
# Onderstaande coefficienten komen in plaats daarvan uit een HERIJKTE, Europese versie: Holleman
# (2001, KNMI wetenschappelijk rapport WR-2001-01, "Hail detection using single-polarization
# radar"), een Europese/C-band-herijking van Waldvogel's/Witt's oorspronkelijke Amerikaanse
# S-band-versie. De coefficienten zelf staan niet letterlijk als vergelijking in Hollemans eigen
# rapport (dat rekent i.p.v. daarvan POH=1-FAR uit een empirische FAR-vs-drempel-curve, zie
# verderop) - de exacte vorm "POH(%) = 0.319 + 0.133*ΔH" is teruggevonden in Lukach, Foresti,
# Giot & Delobbe (2017), "Estimating the occurrence and severity of hail based on 10 years of
# observations from weather radar in Belgium", Meteorol. Appl. (hun vgl. 1), expliciet
# toegeschreven aan Hollemans herijking.
#
# AANVULLENDE VERIFICATIE (25 juli 2026): het volledige Holleman (2001) WR-2001-01-rapport zelf
# is ingezien (niet alleen het secundaire Lukach-artikel). Bevestigd: het rapport zet ΔH om naar
# POH via 1-FAR, afgelezen van een empirische FAR-vs-drempel-curve uit hun eigen 1999-
# verificatiedata (hun Fig. 5.2), GEEN lineaire formule. De drie POH-niveaus die het rapport zelf
# noemt voor het semi-operationele KNMI-product van 2000 (25%/65%/85% bij ΔH=-0.3/1.75/4.0km) komen
# niet exact overeen met wat onderstaande lineaire formule op diezelfde punten geeft (27.9%/55.2%/
# 85.1%) - vooral bij ΔH=1.75km ("beste onbevooroordeelde prestatie" volgens het rapport zelf)
# wijkt de lineaire fit ~10 procentpunt af (55.2% vs 65%); bij ΔH=4.0km komt de fit wel vrijwel
# exact uit. Conclusie: de Lukach-vergelijking hieronder is een goede, maar NIET exacte lineaire
# benadering van Hollemans eigen empirische curve - gebruikt hier omdat een closed-form formule
# nodig is i.p.v. een afgeleide curve, met dit kleine nauwkeurigheidsverlies rond het middenbereik
# als bewuste afweging.
POH_SLOPE = 0.133   # POH = POH_INTERCEPT + POH_SLOPE * (H45dBZ_km - H0_km)
POH_INTERCEPT = 0.319


def poh_from_height_diff(h45_km, h0_km):
    """POH (Probability of Hail, fractie 0-1) als functie van het hoogteverschil tussen het
    45dBZ-echo en het 0C-niveau (beide in km ARL/AGL).

    POH = POH_INTERCEPT + POH_SLOPE * (H45dBZ - H0), geclipt tussen 0 en 1. Zie de
    kanttekening bij de module-constanten hierboven: dit is Holleman (2001)'s KNMI-herijking
    (exacte vorm teruggevonden in Lukach et al. 2017), NIET Witt et al. (1998)'s eigen, nooit
    als vergelijking teruggevonden coefficienten - en een lineaire benadering van Hollemans
    eigen, empirisch afgelezen curve (zie de aanvullende verificatie hierboven).

    h45_km: hoogte (km, ARL) van het hoogste punt waar reflectiviteit >=45dBZ is gemeten.
        None/NaN als er geen 45dBZ-echo in de kolom zit (geeft dan POH=0, geen hagelsignaal).
    h0_km: hoogte (km) van het 0C-niveau - zelfde bron als voor SHI/MESH/POSH.

    Werkt op scalars en arrays (met broadcasting), net als de rest van deze module.
    """
    h45_km = np.asarray(h45_km, dtype='float64')
    h0_km = np.asarray(h0_km, dtype='float64')

    diff = h45_km - h0_km
    poh = POH_INTERCEPT + POH_SLOPE * diff
    poh = np.nan_to_num(poh, nan=0.0)
    return np.clip(poh, 0.0, 1.0)


# Koppeling tussen de kalibratie-namen uit gv.mesh_calibration_settings (nlr_globalvars.py) en de
# bijbehorende SHI->mm-functie hierboven, zodat nlr_derived_plain.calculate_MESH() puur op naam kan
# kiezen zonder de drie functies zelf te hoeven kennen (single source of truth voor deze koppeling).
MESH_CALIBRATION_FUNCTIONS = {
    'Witt 1998': mesh_from_shi,
    'Murillo & Homeyer P75': mesh75_from_shi,
    'Murillo & Homeyer P95': mesh95_from_shi,
    # Zelfde eind-formule (2.54*sqrt(SHI)) als 'Witt 1998' - het verschil met deze optie zit
    # NIET in deze stap, maar in de SHI zelf (zie MESH_EDOT_FUNCTIONS hieronder: deze optie
    # gebruikt de ongecorrigeerde Edot-formule, een andere SHI dus).
    'Witt 1998 (S-band, ongecorrigeerd)': mesh_from_shi,
}

# NIEUW (25 juli 2026): koppeling tussen kalibratienaam en de te gebruiken Edot-functie (stap 1
# van de SHI-berekening, zie hail_kinetic_energy_flux hierboven vs. de ongecorrigeerde variant).
# In tegenstelling tot MESH_CALIBRATION_FUNCTIONS hierboven (die alleen de LAATSTE stap SHI->mm
# verandert) verandert deze koppeling de SHI ZELF. 'Witt 1998'/P75/P95 delen dezelfde
# (C-band-gecorrigeerde) SHI - Murillo & Homeyer (2019) herzagen alleen de laatste omzetstap,
# niet de onderliggende energieformule. Alleen de nieuwe 'ongecorrigeerde' optie wijkt hier af.
MESH_EDOT_FUNCTIONS = {
    'Witt 1998': hail_kinetic_energy_flux,
    'Murillo & Homeyer P75': hail_kinetic_energy_flux,
    'Murillo & Homeyer P95': hail_kinetic_energy_flux,
    'Witt 1998 (S-band, ongecorrigeerd)': hail_kinetic_energy_flux_uncorrected_Sband,
}


if __name__ == '__main__':
    print("Zelftest: reflectivity_weight (W(Z))")
    print("-" * 70)
    test_Z = np.array([20.0, 40.0, 45.0, 50.0, 65.0])
    w = reflectivity_weight(test_Z)
    print(f"  Z={test_Z} -> W(Z)={w}")
    assert w[0] == 0.0, "onder 40 dBZ moet W(Z)=0 zijn"
    assert w[1] == 0.0, "op 40 dBZ moet W(Z)=0 zijn (ondergrens)"
    assert abs(w[2] - 0.5) < 1e-9, "op 45 dBZ (midden 40-50) moet W(Z)=0.5 zijn"
    assert w[3] == 1.0, "op 50 dBZ moet W(Z)=1 zijn (bovengrens)"
    assert w[4] == 1.0, "boven 50 dBZ moet W(Z)=1 blijven"
    print("  OK\n")

    print("Zelftest: height_weight (W_T(h))")
    print("-" * 70)
    h0, hm20 = 3000.0, 6000.0
    test_h = np.array([2000.0, 3000.0, 4500.0, 6000.0, 8000.0])
    wt = height_weight(test_h, h0, hm20)
    print(f"  h={test_h} (h0={h0}, h-20={hm20}) -> W_T(h)={wt}")
    assert wt[0] == 0.0, "onder h0 moet W_T=0 zijn"
    assert wt[1] == 0.0, "op h0 moet W_T=0 zijn"
    assert abs(wt[2] - 0.5) < 1e-9, "op het midden moet W_T=0.5 zijn"
    assert wt[3] == 1.0, "op h-20 moet W_T=1 zijn"
    assert wt[4] == 1.0, "boven h-20 moet W_T=1 blijven"
    print("  OK\n")

    print("Zelftest: hail_kinetic_energy_flux (Edot)")
    print("-" * 70)
    for z in (40.0, 50.0, 60.0, 65.0):
        e = hail_kinetic_energy_flux(np.array([z]))[0]
        print(f"  Z={z} dBZ -> Edot={e:.3f} J/m^2/s")
    e_nan = hail_kinetic_energy_flux(np.array([np.nan]))[0]
    assert e_nan == 0.0, "ontbrekende Z moet Edot=0 geven (draagt niets bij)"
    print("  OK (ontbrekende Z geeft terecht Edot=0)\n")

    print("Zelftest: hail_kinetic_energy_flux_uncorrected_Sband, vergeleken met de C-band-gecorrigeerde variant")
    print("-" * 70)
    for z in (40.0, 50.0, 60.0, 65.0):
        e_corr = hail_kinetic_energy_flux(np.array([z]))[0]
        e_uncorr = hail_kinetic_energy_flux_uncorrected_Sband(np.array([z]))[0]
        print(f"  Z={z} dBZ -> gecorrigeerd={e_corr:.4f}, ongecorrigeerd(S-band)={e_uncorr:.4f}")
        assert e_corr > e_uncorr, "gecorrigeerde (C-band) Edot moet hoger zijn dan de kale S-band-formule (zie module-docstring bij hail_kinetic_energy_flux_uncorrected_Sband)"
    e_nan_uncorr = hail_kinetic_energy_flux_uncorrected_Sband(np.array([np.nan]))[0]
    assert e_nan_uncorr == 0.0, "ontbrekende Z moet ook bij de ongecorrigeerde variant Edot=0 geven"
    assert MESH_EDOT_FUNCTIONS['Witt 1998 (S-band, ongecorrigeerd)'] is hail_kinetic_energy_flux_uncorrected_Sband
    assert MESH_EDOT_FUNCTIONS['Witt 1998'] is hail_kinetic_energy_flux
    print("  OK (gecorrigeerd > ongecorrigeerd bij elke Z, dict-koppeling klopt)\n")

    print("Zelftest: severe_hail_index + mesh_from_shi, met een synthetisch hagelprofiel")
    print("-" * 70)
    # Synthetisch profiel: een storm met een kern van 65 dBZ die ver boven -20C reikt
    # (een typisch "diepe hagelkern"-scenario), h0=3000m, h-20=6500m.
    heights = np.arange(0, 12001, 250.0)  # 0 tot 12 km, elke 250m
    # Z neemt toe tot 65 dBZ rond 5-7km, en neemt daarna weer af richting de storm-top.
    Z_profile = 65.0 - 0.004 * (heights - 6000.0) ** 2 / 1000.0
    Z_profile = np.clip(Z_profile, -30, 65)  # ondergrens zodat het geen extreem negatieve dBZ wordt
    h0_m, h_minus20_m = 3000.0, 6500.0

    shi = severe_hail_index(Z_profile, heights, h0_m, h_minus20_m)
    mesh = mesh_from_shi(shi)
    print(f"  Diepe hagelkern (piek 65 dBZ rond 6km, h0=3000m, h-20=6500m):")
    print(f"    SHI = {shi:.2f}, MESH = {mesh:.1f} mm")
    assert mesh > 0, "een duidelijke hagelkern zou een positieve MESH moeten geven"

    # Vergelijk met een zwakkere storm (max 45 dBZ) - moet een duidelijk lagere MESH geven.
    Z_profile_weak = 45.0 - 0.004 * (heights - 5000.0) ** 2 / 1000.0
    Z_profile_weak = np.clip(Z_profile_weak, -30, 45)
    shi_weak = severe_hail_index(Z_profile_weak, heights, h0_m, h_minus20_m)
    mesh_weak = mesh_from_shi(shi_weak)
    print(f"  Zwakkere storm (piek 45 dBZ rond 5km): SHI = {shi_weak:.2f}, MESH = {mesh_weak:.1f} mm")
    assert mesh_weak < mesh, "een zwakkere storm moet een lagere MESH geven dan de hagelkern hierboven"
    print("  OK (sterkere kern geeft hogere MESH, zoals verwacht)\n")

    # Test: kolom zonder enige data boven het 0C-niveau (bv. buiten het scanbereik) -> MESH=0
    Z_profile_nodata = np.full_like(heights, np.nan)
    shi_nodata = severe_hail_index(Z_profile_nodata, heights, h0_m, h_minus20_m)
    mesh_nodata = mesh_from_shi(shi_nodata)
    print(f"  Kolom zonder data: SHI = {shi_nodata}, MESH = {mesh_nodata} mm")
    assert shi_nodata == 0.0 and mesh_nodata == 0.0
    print("  OK (kolom zonder data geeft terecht MESH=0, geen crash)\n")

    # Test: h0/h-20 ontbreken (None) -> moet netjes 0.0 geven, niet crashen
    shi_notemp = severe_hail_index(Z_profile, heights, None, None)
    print(f"  h0/h-20 ontbreken (None): SHI = {shi_notemp}")
    assert shi_notemp == 0.0
    print("  OK (ontbrekende temperatuurdata geeft terecht SHI=0, geen crash)\n")

    print("Zelftest: warning_threshold + posh_from_shi")
    print("-" * 70)
    h0_test_m = 3000.0  # zelfde 0C-hoogte als het hagelkern-scenario hierboven
    wt = warning_threshold(h0_test_m)
    print(f"  H0={h0_test_m}m -> WT={wt:.2f}")
    assert wt >= POSH_WT_MIN, "WT mag nooit onder de ondergrens zakken"

    posh_strong = posh_from_shi(shi, h0_test_m)
    posh_weak = posh_from_shi(shi_weak, h0_test_m)
    print(f"  Diepe hagelkern: SHI={shi:.2f} -> POSH={posh_strong:.1f}%")
    print(f"  Zwakkere storm:  SHI={shi_weak:.2f} -> POSH={posh_weak:.1f}%")
    assert 0.0 <= posh_strong <= 100.0 and 0.0 <= posh_weak <= 100.0, "POSH moet altijd tussen 0-100% liggen"
    assert posh_strong >= posh_weak, "een sterkere hagelkern moet een gelijke of hogere POSH geven"

    posh_nodata = posh_from_shi(0.0, h0_test_m)
    print(f"  SHI=0 (geen hagelsignaal): POSH={posh_nodata}%")
    assert posh_nodata == 0.0, "SHI=0 moet POSH=0 geven, geen crash op ln(0)"
    print("  OK (POSH geclipt tussen 0-100%, consistent met MESH-sterkte)\n")

    print("Zelftest: mesh75_from_shi + mesh95_from_shi (Murillo & Homeyer 2019)")
    print("-" * 70)
    mesh75_strong = mesh75_from_shi(shi)
    mesh95_strong = mesh95_from_shi(shi)
    mesh75_weak = mesh75_from_shi(shi_weak)
    mesh95_weak = mesh95_from_shi(shi_weak)
    print(f"  Diepe hagelkern: SHI={shi:.2f} -> MESH75={mesh75_strong:.1f}mm, MESH95={mesh95_strong:.1f}mm (Witt: {mesh:.1f}mm)")
    print(f"  Zwakkere storm:  SHI={shi_weak:.2f} -> MESH75={mesh75_weak:.1f}mm, MESH95={mesh95_weak:.1f}mm (Witt: {mesh_weak:.1f}mm)")
    assert mesh75_strong > mesh75_weak and mesh95_strong > mesh95_weak, "sterkere kern moet hogere MESH75/MESH95 geven"
    # Met de gecorrigeerde coëfficiënten (25 juli 2026) is MESH95 > MESH75 over vrijwel het hele
    # bereik (i.t.t. de eerdere, foute coëfficiënten, die elkaar rond SHI=1 kruisten).
    mesh75_hi = mesh75_from_shi(50.0)
    mesh95_hi = mesh95_from_shi(50.0)
    print(f"  Realistische SHI=50: MESH75={mesh75_hi:.1f}mm, MESH95={mesh95_hi:.1f}mm")
    assert mesh95_hi > mesh75_hi, "bij praktisch relevante SHI (>1) moet MESH95 conservatiever (hoger) zijn dan MESH75"
    mesh75_zero = mesh75_from_shi(0.0)
    mesh95_zero = mesh95_from_shi(0.0)
    print(f"  SHI=0: MESH75={mesh75_zero:.1f}mm, MESH95={mesh95_zero:.1f}mm")
    assert mesh75_zero == 0.0 and mesh95_zero == 0.0, "SHI=0 moet MESH75/MESH95=0 geven, geen crash op 0^exponent"
    print("  OK (MESH75/MESH95 gedragen zich consistent met MESH_Witt)\n")

    print("Zelftest: poh_from_height_diff (Holleman 2001 / Waldvogel-herijking)")
    print("-" * 70)
    poh_none = poh_from_height_diff(0.5, 3.0)  # 45dBZ ONDER het 0C-niveau -> geen hagelsignaal
    poh_marginal = poh_from_height_diff(3.5, 3.0)  # net boven het 0C-niveau
    poh_strong = poh_from_height_diff(5.5, 3.0)  # ruim boven het 0C-niveau
    print(f"  45dBZ 2.5km onder 0C: POH={poh_none*100:.0f}%")
    print(f"  45dBZ 0.5km boven 0C: POH={poh_marginal*100:.0f}%")
    print(f"  45dBZ 2.5km boven 0C: POH={poh_strong*100:.0f}%")
    assert poh_strong > poh_marginal >= 0.0, "hoger 45dBZ-echo boven het 0C-niveau moet een hogere POH geven"
    assert 0.0 <= poh_none <= 1.0 and 0.0 <= poh_strong <= 1.0, "POH moet altijd tussen 0 en 1 liggen"
    poh_nan = poh_from_height_diff(np.nan, 3.0)
    print(f"  Ontbrekend 45dBZ-echo (NaN): POH={poh_nan}")
    assert poh_nan == 0.0, "ontbrekend 45dBZ-echo moet POH=0 geven, geen crash"
    print("  OK (POH stijgt met hoogteverschil, blijft binnen 0-1, geen crash bij ontbrekende data)\n")

    print("ALLE TESTS GESLAAGD")
