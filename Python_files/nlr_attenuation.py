# Copyright (C) 2026
# ZPHI zelfconsistente verzwakkingscorrectie voor C-band, t.b.v. de hagelproducten
# (MESH/SHI/POSH/POH/HCLASS) in NLradar.
#
# Bron, met exacte vergelijkingsnummers (geverifieerd tegen de aangeleverde PDF's):
# - Testud, Le Bouar, Obligis & Ali-Mehenni (2000), "The rain profiling algorithm
#   applied to polarimetric weather radar", J. Atmos. Ocean. Technol. 17:332-356.
# - Bringi, Keenan & Chandrasekar (2001), "Correcting C-band radar reflectivity and
#   differential reflectivity data for rain attenuation: a self-consistent method
#   with constraints", IEEE TGRS 39:1906-1915.
# - Gou, Chen & Zheng (2019), "An improved self-consistent approach to attenuation
#   correction for C-band polarimetric radar measurements...", Atmospheric Research
#   226:32-48. Vergelijkingen (2a)-(2g) voor Z, (3a)-(3e) voor ZDR, en de 4 extra
#   randvoorwaarden uit hun sectie 2.2 zijn hieronder rechtstreeks geimplementeerd.
#
# Z-CORRECTIE (Gou 2019, vgl. 2a-2g), per segment [r0,rm] (aaneengesloten neerslag):
#   Ah(r)   = Zm(r)^b * [10^(0.1*b*alpha*dPhi) - 1] / (I(r0,rm) + [10^(0.1*b*alpha*dPhi)-1]*I(r,rm))
#   dPhi    = PhiDP_filtered(rm) - PhiDP_filtered(r0)                              (2b)
#   I(r0,rm)= 0.46*b * integraal[r0,rm] Zm(s)^b ds                                 (2c)
#   I(r,rm) = 0.46*b * integraal[r,rm]  Zm(s)^b ds                                 (2d)
#   kostenfunctie(alpha) = integraal[r0,rm] |PhiDP_rec(s,alpha) - PhiDP_filtered(s)| ds   (2e)
#     met PhiDP_rec(r0,rm) = integraal[r0,rm] Ah(s,alpha)/alpha ds                 (2f)
#   -> alpha_opt = alpha uit {0.03,...,0.18 stap 0.01} die de kostenfunctie minimaliseert
#   Zh_corrected(r) = Zh_measured(r) + 2*integraal[0,r] Ah(s,alpha_opt) ds          (2g)
#
# Extra randvoorwaarden (Gou 2019, sectie 2.2), (i)-(iii) hieronder geimplementeerd:
#   (i)   Ah niet-negatief: alleen segmenten met monotoon stijgende PhiDP (dPhi>=drempel)
#   (ii)  RhoHV-segmentatie: aaneengesloten gates met RhoHV>=0.85 vormen een segment
#         (0.98-grens voor de rain/mixed-phase-onderverdeling BINNEN zo'n segment is
#         voor nu NIET apart doorgevoerd - zie opmerking onderaan dit bestand)
#   (iii) Convergentie-eis: als het gereconstrueerde eindpunt PhiDP_rec(rm) het gemeten
#         dPhi ruim overschrijdt (zelfs bij de beste alpha), wordt het segment op de
#         helft gesplitst (split-half) en apart opnieuw verwerkt (max_split_depth keer)
#
# ZDR-CORRECTIE (Gou 2019, vgl. 3a-3e): hier is BEWUST alleen vergelijking 3a
# geimplementeerd (Bringi 2001's algemene lineaire ZDR-ZH-relatie, dBZ-schaal), NIET
# vergelijking 3b (Gou's exponentiele relatie) - die is expliciet afgeleid uit lokale
# druppelgrootteverdeling-metingen rond Hangzhou en dus site-specifiek, niet zomaar
# toepasbaar op Nederlandse C-band-data zonder eigen DSD-kalibratie.
#   Zdr_hat(r) = 0                        als Zh_corrected(r) < 20 dBZ             (3a)
#   Zdr_hat(r) = 0.048*Zh_corrected(r)-0.774   als 20<=Zh_corrected(r)<=45 dBZ     (3a)
#   Adp(r;beta) = (beta/alpha_opt) * Ah(r;alpha_opt)                              (3c)
#   Zdr_corrected(r;beta) = Zdr_measured(r) + 2*integraal[0,r] Adp(s;beta) ds      (3d)/(3e)
# Omdat Zdr_corrected(r;beta) LINEAIR is in beta (Adp schaalt lineair met beta), is de
# iteratieve zoekprocedure uit het artikel wiskundig gelijk aan 1 kleinste-kwadraten-
# schatting van beta die Zdr_corrected(r;beta) zo dicht mogelijk bij Zdr_hat(r) brengt
# over het hele segment - dat is hier direct (niet-iteratief) opgelost, met hetzelfde
# resultaat.

import numpy as np
from scipy.ndimage import uniform_filter1d


def unwrap_phidp_deg(phidp_deg):
    """Ontwikkelt (unwrap) een PhiDP-array (graden, per radiaal langs de laatste as)
    die door de 0/360-grens kan wikkelen, zoals waargenomen in de echte Herwijnen-
    testdata (bv. 355.49 -> 356.13 -> 0.93 -> 12.51 graden binnen een en dezelfde
    radiaal).
    """
    rad = np.deg2rad(phidp_deg)
    unwrapped_rad = np.unwrap(rad, axis=-1)
    return np.rad2deg(unwrapped_rad)


def despike_and_smooth_phidp(phidp_unwrapped_deg, spike_window=4, spike_threshold_deg=45.0,
                              smooth_window=9):
    """Voorbewerking van het (reeds ontwikkelde) PhiDP-profiel, zoals beschreven in Gou et
    al. (2019) sectie 3 ("ΨDP processing"). Werkt op elke vorm array, langs de LAATSTE as
    (dus zowel op 1 radiaal (1D) als een hele scan tegelijk (2D, azimuth x range) - bij een
    2D-array gebeurt dit voor alle radialen in 1 vectorbewerking, wat essentieel is voor de
    snelheid: de eerdere per-gate Python-lus (1x per radiaal aangeroepen) bleek in de
    praktijk te traag (2,8s per scan, oplopend tot ~3 minuten voor een heel volume over
    meerdere hagelproducten - Erik meldde dit als "NLradar loopt vast", 28 juli 2026).

    Twee stappen:
    1. Spike-onderdrukking: elke gate wordt vergeleken met het voortschrijdend gemiddelde
       over een venster van (2*spike_window+1) gates eromheen (inclusief zichzelf - een
       lichte vereenvoudiging t.o.v. "de 4 dichtstbijzijnde gates exclusief zichzelf" uit
       het artikel, met verwaarloosbaar effect bij window>=9). Wijkt een gate meer dan
       spike_threshold_deg (het artikel: 45 graden) af van dat lokale gemiddelde, dan wordt
       de gate vervangen door dat gemiddelde.
    2. Gladstrijken (substituut voor het artikel's FIR-filter, dat zelf niet met exacte
       coefficienten gespecificeerd is in de tekst): een voortschrijdend-gemiddelde-filter
       (boxcar) over 'smooth_window' gates.

    Beide stappen zijn hier volledig gevectoriseerd via scipy.ndimage.uniform_filter1d
    (werkt native in C, geen Python-lus over gates of radialen).
    """
    arr = np.asarray(phidp_unwrapped_deg, dtype='float64')
    n = arr.shape[-1]

    spike_win_len = min(2 * spike_window + 1, n)
    local_mean = uniform_filter1d(arr, size=spike_win_len, axis=-1, mode='nearest')
    despike = np.where(np.abs(arr - local_mean) > spike_threshold_deg, local_mean, arr)

    if smooth_window < 2 or n < smooth_window:
        return despike
    smoothed = uniform_filter1d(despike, size=min(smooth_window, n), axis=-1, mode='nearest')
    return smoothed


def _segment_bounds(rhohv, threshold, min_gates):
    """Vindt aaneengesloten segmenten met RhoHV >= threshold en minimale lengte
    min_gates. Retourneert een lijst van (start,eind)-indices (eind exclusief).
    """
    good = rhohv >= threshold
    segments = []
    start = None
    for i, g in enumerate(good):
        if g and start is None:
            start = i
        elif not g and start is not None:
            if i - start >= min_gates:
                segments.append((start, i))
            start = None
    if start is not None and len(good) - start >= min_gates:
        segments.append((start, len(good)))
    return segments


def _cumulative_I(Zlin_b, range_res_km, b):
    """I(r,rm) = 0.46*b*integraal[r,rm] Z(s)^b ds, als achterwaartse cumulatieve
    trapeziumsom (index 0 = r0, laatste index = rm). Retourneert een array van
    dezelfde lengte, I_r_rm[k] = I(r_k, rm).
    """
    n = len(Zlin_b)
    I_r_rm = np.zeros(n, dtype='float64')
    if n > 1:
        trap = 0.5 * (Zlin_b[:-1] + Zlin_b[1:]) * range_res_km
        I_r_rm[:-1] = np.cumsum(trap[::-1])[::-1]
    return 0.46 * b * I_r_rm


def _solve_segment(Zlin_b, phidp_filtered_seg, range_res_km, b, alphas,
                    fallback_alpha=0.105, boundary_tol=1e-9):
    """Lost een enkel segment op volgens vgl. (2a)-(2f). Retourneert (Ah_seg,
    alpha_opt, phidp_rec_end, converged), waarbij converged=False betekent dat
    zelfs bij de beste alpha de kostenfunctie abnormaal convergeerde (vgl. (iii):
    PhiDP_rec(rm) overschrijdt het gemeten dPhi ruim) en het segment gesplitst
    zou moeten worden.

    TERUGVALOPTIE (sessie-overleg 28 juli): als de gezochte alpha exact op de rand
    van het bereik (alpha_min of alpha_max) uitkomt, is dat een teken dat de
    kostenfunctie (2e) geen echt intern minimum heeft gevonden voor dit segment
    (empirisch bevestigd: op echte Herwijnen-data ~95% randgevallen). In dat geval
    wordt Ah in plaats daarvan berekend met een vaste, neutrale alpha=fallback_alpha
    (standaard 0.105, het midden van [0.03,0.18]) - dezelfde vgl. (2a)-formule, alleen
    zonder de (hier onbetrouwbaar gebleken) optimalisatie. Waar de zoektocht wel een
    interne waarde vindt (een echt signaal dat er een minimum is), blijft die waarde
    gewoon gebruikt. fallback_alpha=0.105 komt nooit voor in het gezochte rooster
    (stappen van 0.01 vanaf 0.03), dus is achteraf herkenbaar in alpha_used.

    SNELHEID (21 september 2026): de alpha-zoektocht zelf (16 waarden) bleek de
    bottleneck bij bladeren door MESH/POH/SHI/POSH - deze functie draait per
    RhoHV-segment, per radiaal (~360), per scan (~14-16), per tijdstap, en werd
    voorheen met een Python-lus over de 16 alpha-waarden uitgevoerd. Hieronder is
    die lus vervangen door 1 gevectoriseerde berekening over alle 16 alpha-waarden
    tegelijk (extra as, shape (16, segmentlengte)) - exact dezelfde formules en
    exact dezelfde selectielogica (eerste alpha bij gelijke kosten wint, net als
    de oorspronkelijke strikte "<"-vergelijking), dus bit-identieke uitkomst,
    alleen met veel minder Python-overhead per segment. Metingen bleven uit
    (geen productieomgeving hier beschikbaar) - wel geverifieerd dat elke
    if/continue-voorwaarde uit de oude lus hieronder zijn exacte vectoriseerde
    tegenhanger heeft (zie de toelichting per blok).
    """
    seg_len = len(Zlin_b)
    dPhi = phidp_filtered_seg[-1] - phidp_filtered_seg[0]
    I_r_rm = _cumulative_I(Zlin_b, range_res_km, b)
    I_r0_rm = I_r_rm[0]

    if dPhi <= 0 or I_r0_rm <= 0:
        return np.zeros(seg_len), None, None, True

    alphas_arr = np.asarray(alphas, dtype='float64')  # shape (K,)
    K = len(alphas_arr)

    # vgl. (2a)-blok, alle K alpha's tegelijk: C heeft shape (K,), denom/Ah_all
    # shape (K, seg_len) via broadcasting (alpha-as vooraan, gate-as achteraan).
    C = 10. ** (0.1 * b * alphas_arr * dPhi) - 1.  # shape (K,)
    valid_C = C > 0  # was: "if C <= 0: continue"

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        denom = I_r0_rm + C[:, None] * I_r_rm[None, :]  # (K, seg_len)
        # was: "if np.any(denom <= 0): continue" -> geldig vereist ALLE gates > 0
        valid_denom = valid_C & np.all(denom > 0, axis=1)
        denom_safe = np.where(denom > 0, denom, 1.)  # voorkomt deling door 0 voor ongeldige rijen
        Ah_all = Zlin_b[None, :] * C[:, None] / denom_safe  # (K, seg_len)

    # was: "if np.any(~np.isfinite(Ah_seg)) or np.any(Ah_seg < 0): continue" ->
    # geldig vereist ALLE gates eindig EN >= 0.
    finite_ok = np.all(np.isfinite(Ah_all), axis=1) & np.all(Ah_all >= 0, axis=1)
    valid = valid_denom & finite_ok

    if not np.any(valid):
        return np.zeros(seg_len), None, None, True

    # PhiDP_rec(r) = integraal[r0,r] Ah(s,alpha)/alpha ds (vgl. 2f) - per alpha-rij
    # dezelfde trapeziumsom als voorheen, nu voor alle K rijen tegelijk.
    phidp_rec = np.zeros((K, seg_len), dtype='float64')
    if seg_len > 1:
        with np.errstate(invalid='ignore'):
            trap = 0.5 * (Ah_all[:, :-1] + Ah_all[:, 1:]) * range_res_km  # (K, seg_len-1)
            phidp_rec[:, 1:] = np.cumsum(trap, axis=1) / alphas_arr[:, None]

    # Kostenfunctie (vgl. 2e), voor alle K alpha's tegelijk.
    phidp_filtered_rel = phidp_filtered_seg - phidp_filtered_seg[0]
    trapz_fn = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    with np.errstate(invalid='ignore'):
        cost = trapz_fn(np.abs(phidp_rec - phidp_filtered_rel[None, :]), dx=range_res_km, axis=1)  # (K,)

    # was: "if cost < best_cost" (strikt kleiner dan, dus bij gelijke kosten wint de
    # EERSTE alpha in de volgorde) -> np.argmin geeft ook de eerste laagste waarde
    # bij gelijke kosten, dus zelfde tie-break-gedrag. Ongeldige rijen krijgen +inf
    # zodat ze nooit gekozen worden, ook als hun (ongebruikte) cost toevallig NaN is.
    cost_masked = np.where(valid, cost, np.inf)
    best_idx = int(np.argmin(cost_masked))
    if not np.isfinite(cost_masked[best_idx]):
        return np.zeros(seg_len), None, None, True

    best_alpha = alphas_arr[best_idx]
    best_Ah = Ah_all[best_idx]
    best_phidp_rec = phidp_rec[best_idx, -1]

    # Terugvaloptie: alpha op de rand van het gezochte bereik -> onbetrouwbaar, gebruik
    # de vaste neutrale waarde i.p.v. de randwaarde (zelfde vgl. 2a-formule).
    at_boundary = (abs(best_alpha - alphas_arr[0]) < boundary_tol) or (abs(best_alpha - alphas_arr[-1]) < boundary_tol)
    if at_boundary:
        alpha = fallback_alpha
        C = 10. ** (0.1 * b * alpha * dPhi) - 1.
        denom = I_r0_rm + C * I_r_rm
        Ah_fallback = Zlin_b * C / denom
        phidp_rec_fb = np.zeros(seg_len, dtype='float64')
        if seg_len > 1:
            trap = 0.5 * (Ah_fallback[:-1] + Ah_fallback[1:]) * range_res_km
            phidp_rec_fb[1:] = np.cumsum(trap) / alpha
        best_alpha = fallback_alpha
        best_Ah = Ah_fallback
        best_phidp_rec = phidp_rec_fb[-1]

    # Randvoorwaarde (iii): convergentie-eis. Als het gereconstrueerde PhiDP-eindpunt
    # het gemeten dPhi met meer dan 20% (of >5 graden) overschrijdt, abnormale convergentie.
    converged = best_phidp_rec <= dPhi * 1.2 + 5.0
    return best_Ah, best_alpha, best_phidp_rec, converged


def correct_ray_zphi(Z_dBZ, PhiDP_deg, RhoHV, range_res_km,
                      b=0.78, alpha_min=0.03, alpha_max=0.18, alpha_step=0.01,
                      rhohv_threshold=0.85, min_delta_phidp_deg=2.0, min_segment_gates=5,
                      max_total_correction_dB=15.0, max_split_depth=3,
                      phidp_already_smoothed=None):
    """Corrigeert een enkele radiaal (1D-arrays, gate 0 = dichtst bij de radar) volgens
    de ZPHI-methode (Gou 2019, vgl. 2a-2g). Retourneert (Z_corrected, Ah_profile,
    alpha_used_per_gate). Waar geen correctie kon worden bepaald blijft Ah=0.

    phidp_already_smoothed: optioneel, een al ontwikkeld+gladgestreken PhiDP-profiel
    (zie despike_and_smooth_phidp) - gebruikt door correct_scan_zphi, dat deze stap 1x
    voor de hele scan vectoriseert i.p.v. hem hier opnieuw per radiaal te herhalen (dat
    bleek in de praktijk de snelheidsbottleneck, zie toelichting bij despike_and_smooth_phidp).
    Bij los gebruik van deze functie (1 radiaal) wordt PhiDP_deg gewoon zelf verwerkt.
    """
    n = len(Z_dBZ)
    Ah = np.zeros(n, dtype='float64')
    alpha_used = np.full(n, np.nan, dtype='float64')

    if phidp_already_smoothed is not None:
        phidp_unwrapped = phidp_already_smoothed
    else:
        phidp_unwrapped_raw = unwrap_phidp_deg(PhiDP_deg)
        phidp_unwrapped = despike_and_smooth_phidp(phidp_unwrapped_raw)
    Zlin_b = np.power(10., 0.1 * b * np.asarray(Z_dBZ, dtype='float64'))
    alphas = np.arange(alpha_min, alpha_max + 1e-9, alpha_step)

    top_segments = _segment_bounds(RhoHV, rhohv_threshold, min_segment_gates)


    def process(s, e, depth):
        seg_len = e - s
        if seg_len < min_segment_gates:
            return
        dPhi_check = phidp_unwrapped[e - 1] - phidp_unwrapped[s]
        if not np.isfinite(dPhi_check) or dPhi_check < min_delta_phidp_deg:
            return  # randvoorwaarde (i): geen bruikbare (monotoon stijgende) fase-opbouw
        Ah_seg, alpha_opt, _, converged = _solve_segment(
            Zlin_b[s:e], phidp_unwrapped[s:e], range_res_km, b, alphas)
        if alpha_opt is None:
            return
        if not converged and depth < max_split_depth and seg_len >= 2 * min_segment_gates:
            mid = s + seg_len // 2
            process(s, mid, depth + 1)
            process(mid, e, depth + 1)
            return
        Ah[s:e] = Ah_seg
        alpha_used[s:e] = alpha_opt

    for (s, e) in top_segments:
        process(s, e, 0)

    PIA = 2. * range_res_km * np.cumsum(Ah)
    PIA = np.minimum(PIA, max_total_correction_dB)
    Z_corrected = np.asarray(Z_dBZ, dtype='float64') + PIA
    return Z_corrected.astype('float32'), Ah.astype('float32'), alpha_used


def correct_ray_zdr_zphi(Z_corrected_dBZ, ZDR_dBZ, Ah_profile, alpha_used, range_res_km):
    """Corrigeert ZDR voor een enkele radiaal volgens Gou 2019 vgl. (3a),(3c)-(3e),
    gegeven het reeds berekende Ah-profiel en de gebruikte alpha uit de Z-correctie
    (correct_ray_zphi hierboven). Gebruikt de lineaire Bringi(2001)-ZDR-ZH-relatie
    (vgl. 3a), NIET Gou's Hangzhou-specifieke exponentiele variant (vgl. 3b) - zie
    de toelichting bovenaan dit bestand.

    Retourneert ZDR_corrected. Segmenten zonder geldige alpha (Ah=0/alpha=nan)
    blijven ongewijzigd.
    """
    n = len(ZDR_dBZ)
    ZDR_corrected = np.array(ZDR_dBZ, dtype='float64', copy=True)

    valid = np.isfinite(alpha_used) & (Ah_profile > 0)
    if not np.any(valid):
        return ZDR_corrected.astype('float32')

    idx = np.where(valid)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    seg_starts = np.concatenate(([0], breaks + 1))
    seg_ends = np.concatenate((breaks + 1, [len(idx)]))

    for ss, se in zip(seg_starts, seg_ends):
        s, e = idx[ss], idx[se - 1] + 1
        alpha_opt = alpha_used[s]  # constant binnen het segment
        Ah_seg = Ah_profile[s:e]

        Zh_c = Z_corrected_dBZ[s:e]
        Zdr_hat = np.where(Zh_c < 20., 0., np.clip(0.048 * Zh_c - 0.774, 0., None))
        Zdr_hat = np.where(Zh_c > 45., 0.048 * 45. - 0.774, Zdr_hat)

        seg_len = e - s
        PIA_H = np.zeros(seg_len, dtype='float64')
        if seg_len > 1:
            trap = 0.5 * (Ah_seg[:-1] + Ah_seg[1:]) * range_res_km
            PIA_H[1:] = 2. * np.cumsum(trap)

        Zdr_meas = ZDR_dBZ[s:e]
        x = PIA_H / alpha_opt
        target = Zdr_hat - Zdr_meas
        denom = np.sum(x * x)
        if denom <= 0:
            continue
        beta_opt = np.sum(x * target) / denom
        beta_opt = np.clip(beta_opt, 0., 1.)

        ZDR_corrected[s:e] = Zdr_meas + beta_opt * x

    return ZDR_corrected.astype('float32')


def correct_scan_zphi(Z_dBZ_2d, PhiDP_deg_2d, RhoHV_2d, range_res_km,
                       ZDR_dBZ_2d=None, **kwargs):
    """Corrigeert een volledige scan (azimuth x range). Retourneert Z_corrected_2d,
    en (als ZDR_dBZ_2d is meegegeven) ook ZDR_corrected_2d.

    Snelheid (28 juli 2026): de PhiDP-voorbewerking (unwrap+despike+smooth) gebeurt hier
    1x voor de HELE scan tegelijk (gevectoriseerd over alle radialen), i.p.v. opnieuw per
    radiaal binnen correct_ray_zphi - dat laatste bleek in de praktijk te traag (2,8s per
    scan, opliep tot een paar minuten per volume - door Erik gemeld als "NLradar loopt
    vast"). Het overgebleven per-radiaal-werk (segmentatie + alpha-zoektocht) is per
    radiaal inherent verschillend (variabele segmentgrenzen) en blijft daarom een
    Python-lus, maar is zelf niet de bottleneck gebleken.

    NaN-BUGFIX (28 juli 2026, Erik meldde: correctie leek geen effect te hebben - bleek
    dat alle diff-metingen NaN opleverden): NLradar markeert data buiten het bruikbare
    bereik (clutter/geen signaal) als NaN, niet als een sentinelwaarde. Zonder correctie
    hiervoor lekt 1 enkele NaN-gate via de cumulatieve sommen (voor zowel de PhiDP-
    gladstrijking als de Ah/PIA-integratie) door naar ALLE gates erna in diezelfde
    radiaal, en corrumpeert zo de hele correctie. Fix: elke gate waar Z, PhiDP of RhoHV
    NaN is, wordt vooraf als "ongeldig" gemarkeerd (RhoHV->0, altijd buiten elk segment,
    consistent met hoe echt slechte RhoHV al werd behandeld) en na afloop weer expliciet
    op NaN gezet in de uitvoer, zodat de oorspronkelijke maskering intact blijft.
    """
    Z_arr = np.asarray(Z_dBZ_2d, dtype='float64')
    PhiDP_arr = np.asarray(PhiDP_deg_2d, dtype='float64')
    RhoHV_arr = np.asarray(RhoHV_2d, dtype='float64')
    invalid = ~np.isfinite(Z_arr) | ~np.isfinite(PhiDP_arr) | ~np.isfinite(RhoHV_arr)

    Z_clean = np.where(invalid, -30., Z_arr)
    PhiDP_clean = np.where(invalid, 0., PhiDP_arr)
    RhoHV_clean = np.where(invalid, 0., RhoHV_arr)  # forceert uitsluiting uit elk segment

    phidp_unwrapped_raw = unwrap_phidp_deg(PhiDP_clean)
    phidp_smoothed_2d = despike_and_smooth_phidp(phidp_unwrapped_raw)

    n_az = Z_dBZ_2d.shape[0]
    Z_corrected = np.array(Z_dBZ_2d, dtype='float32', copy=True)
    ZDR_corrected = np.array(ZDR_dBZ_2d, dtype='float32', copy=True) if ZDR_dBZ_2d is not None else None

    for az in range(n_az):
        z_ray, ah_ray, alpha_ray = correct_ray_zphi(
            Z_clean[az], None, RhoHV_clean[az], range_res_km,
            phidp_already_smoothed=phidp_smoothed_2d[az], **kwargs)
        z_ray[invalid[az]] = np.nan  # oorspronkelijke maskering herstellen
        Z_corrected[az] = z_ray
        if ZDR_dBZ_2d is not None:
            zdr_ray = correct_ray_zdr_zphi(
                z_ray, np.where(invalid[az], 0., ZDR_dBZ_2d[az]), ah_ray, alpha_ray, range_res_km)
            zdr_ray[invalid[az]] = np.nan
            ZDR_corrected[az] = zdr_ray

    if ZDR_dBZ_2d is not None:
        return Z_corrected, ZDR_corrected
    return Z_corrected

# LET OP - nog niet geimplementeerd, met opzet (zie ook toelichting bovenaan):
# - De verdere onderverdeling van een RhoHV>=0.85-segment in een apart
#   "puur-regen" (RhoHV>=0.98) en "mixed-phase" (0.85<=RhoHV<0.98) subsegment
#   (Gou 2019 randvoorwaarde (ii)) is hier niet doorgevoerd; er wordt nu 1
#   alpha per aaneengesloten RhoHV>=0.85-segment gezocht, niet apart per fase.
# - Gou's Hangzhou-specifieke exponentiele ZDR-ZH-relatie (vgl. 3b) is bewust
#   niet gebruikt (zie toelichting bovenaan) - alleen Bringi's algemene
#   lineaire relatie (vgl. 3a).
