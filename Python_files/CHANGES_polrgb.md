# Polarimetrische RGB composite ('g') voor NLradar (KNMI)

**Update 19**: de tijdsaanduiding in de titel liep niet mee bij PolRGB-panelen (bleef hangen
op een oud tijdstip, terwijl het radarbeeld zelf wel verversle). Tijdens het uitzoeken ook
een tweede, nog niet eerder opgemerkte bug gevonden en gerepareerd (specifiek voor de
IMGW/Polen-radarbron). Alleen `nlr_datasourcegeneral.py` opnieuw aangepast.

## Oorzaak van het tijd-probleem
`_calculate_polrgb` haalt Z/CC/ZDR los op via `get_data_multiple_scans`, die naast de data
ook de bijbehorende scan-tijd teruggeeft — maar die tijd werd tot nu toe gewoon weggegooid
(`data, _, _, _, _ = ...`). Voor elk ander product wordt `self.scantimes[j]` (waar de titel
zijn tijd vandaan haalt) gezet binnen de normale, bron-specifieke `get_data()`-aanroep — die
aanroep slaan we voor 'g' bewust over (zie Update 15/17), dus die tijd-toewijzing gebeurde
voor PolRGB-panelen nergens. De onderliggende data werd wel steeds opnieuw en correct
opgehaald, alleen de getoonde tijd in de titel bleef op het oude, laatst-bekende tijdstip
staan.

## Fix
De scan-tijd van het Z-kanaal (representatief voor alle drie, want ze komen uit dezelfde
scan) wordt nu wel opgevangen en expliciet naar `self.scantimes[j]` geschreven, net als elk
ander product al deed.

## Bonus: latente bug gevonden voor IMGW (Polen)
Tijdens het narekenen bleek dat niet alle radarbronnen evenveel waarden teruggeven vanuit
`get_data_multiple_scans` — de meeste vijf (`data, scantimes, volume_starttime,
volume_endtime, meta`), maar IMGW's eigen formaat (Leonardo Rainbow 3/5) maar vier (zonder
`meta`). De vaste 5-waarden-uitpak-aanroep zou daar een crash hebben gegeven zodra iemand
PolRGB op een IMGW-radar zou proberen — nog niet voorgekomen in onze sessies, maar wel een
reëel, latent probleem. De aanroep is nu flexibel gemaakt zodat die met zowel 4 als 5
retourwaarden werkt.

---

**Update 18**: lettergrootte van de PolRGB-legenda-tekst vergroot. Alleen `nlr_plotting.py`
opnieuw aangepast.

## Lettergrootte
De tick-waarden en labels gebruikten dezelfde (kleine) lettergrootte als de smalle, normale
colorbars (6pt/7pt) — passend voor die context, maar te klein gebleken voor de grotere,
nieuwe PolRGB-balken. Ze hebben nu hun eigen, grotere waarden (9pt/9.8pt — ongeveer 1.4-1.5x
groter), los van de normale colorbars, zodat die laatste niet worden beïnvloed.

## Open item: randdikte links/rechts nog steeds ongelijk
Twee gerichte pogingen (sub-pixel-afronding van de positie; randbreedte gelijktrekken met de
normale, wél-symmetrische colorbars) gaven geen verbetering. Zonder een draaiende
PyQt5/vispy-omgeving om dit zelf te kunnen renderen en inspecteren, kan ik de exacte oorzaak
niet verder isoleren door alleen naar de code te kijken — verdere pogingen zouden op dit
moment giswerk zijn. Dit punt staat open voor een volgende sessie, eventueel met een meer
gerichte diagnostische aanpak (bijvoorbeeld een tijdelijke, sterk vergrote/ingezoomde
weergave om te zien of het verschil consistent in pixels meetbaar is, of varieert met
schermschaling/zoom-niveau).

---

**Update 17**: regressie van de DWD-fix (Update 15) hersteld — PolRGB-panelen toonden geen
data meer, met `KeyError: 0` in de console. Alleen `nlr_datasourcegeneral.py` opnieuw
aangepast.

## Oorzaak (een fout van mijzelf)
De DWD-fix van gisteren sloeg, terecht, de normale bron-specifieke `get_data(j)`-aanroep
over voor product 'g' (omdat die aanroep voor DWD een harde crash gaf, zie Update 15). Maar
die aanroep deed **twee dingen**: data ophalen (overbodig voor 'g', want
`_calculate_polrgb` doet dat zelf al correct) én twee vlaggen zetten
(`self.crd.using_unfilteredproduct[j]`, `self.crd.using_verticalpolarization[j]`) die elders
(`store_data_in_memory`) onvoorwaardelijk worden uitgelezen voor ELK paneel. Door de hele
aanroep over te slaan, werden die twee vlaggen voor 'g'-panelen nooit gezet, wat een
`KeyError` gaf zodra zo'n paneel voor het eerst data probeerde op te slaan — dus eigenlijk
bij elke poging om een PolRGB-paneel te laden.

## Fix
Beide vlaggen worden nu alsnog expliciet op `False` gezet voor product 'g' (een correcte
waarde, want PolRGB heeft geen eigen concept van "ongefilterd product" of "verticale
polarisatie" — dat zijn beide eigenschappen van de onderliggende Z/CC/ZDR-ophaal-stappen
die `_calculate_polrgb` zelf al apart en correct regelt).

## Met excuses
Dit was een eigen fout — bij de DWD-fix van gisteren had ik niet gezien dat de overgeslagen
aanroep een tweede taak had naast het ophalen van data. Bedankt voor het melden met de
volledige console-output, die de oorzaak meteen duidelijk maakte.

---

**Update 16**: PolRGB-legenda herontworpen naar Bram-stijl — drie segmenten onder elkaar in
plaats van naast elkaar, met een tussenstreepje bij bredere schalen. Alleen
`nlr_plotting.py` opnieuw aangepast.

## Wat is veranderd
Op basis van een screenshot van Bram's eigen versie:
- De drie segmenten (Z=rood, ρHV=groen, ZDR=blauw) staan nu **verticaal gestapeld**
  (Z boven, ZDR onder) in plaats van naast elkaar — een smaller, compacter blok.
- Elk segment krijgt een **tussenstreepje met waarde** wanneer zijn bereik (max-min) groter
  is dan 40 — bijvoorbeeld Z's standaardbereik van -10 tot 70 (80 punten) krijgt er een, maar
  ρHV (70-100, 30 punten) en ZDR (0-3, 3 punten) niet, tenzij je die bereiken zelf verbreedt
  via Settings → PolRGB.
- De labels (Z/ρHV/ZDR + eenheid) staan nu rechts van elk segment, op gelijke hoogte met het
  midden van dat segment, in plaats van erboven.

## Nieuwe visual
Er is één nieuwe, simpele `LineVisual` (`polrgb_legend_reflines`) toegevoegd, die de
tussenstreepjes tekent — dit volgt exact hetzelfde patroon als de bestaande
`cbars_reflines`-visual die de normale colorbars ook al gebruiken voor hun eigen
tussenstreepjes.

## Niet aangepast (bewust)
- De positie van de hele legenda blijft zoals die was: vast in de linkerboven-hoek van het
  PolRGB-paneel zelf, niet meeschuivend met het algemene links/rechts-colorbar-systeem.
- De randdikte-asymmetrie die je opmerkte (links iets dikker dan rechts) kon ik niet
  herleiden tot iets in onze eigen code — de balken gebruiken al hetzelfde `ColorBarVisual`-
  type als de normale, al jarenlang werkende colorbars. Het kan een subtiele eigenschap van
  die onderliggende vispy-visual zelf zijn. Laat het weten als dit in de nieuwe, verticale
  layout nog steeds opvalt, dan zoeken we verder.

---

**Update 15**: PolRGB werkte niet bij DWD-radars (toonde "(OLD)", alsof bestanden
ontbraken). Alleen `nlr_datasourcegeneral.py` opnieuw aangepast.

## Oorzaak
Vóór de eigentlijke PolRGB-berekening (`_calculate_polrgb`) roept de algemene `get_data`-
functie altijd eerst de **normale, bron-specifieke** data-ophaalroutine aan
(`self.source_classes[...].get_data(j)`) — bedoeld om losse producten zoals Z, V, CC, ZDR op
te halen. Voor de meeste radarbronnen bleek dat voor product 'g' toevallig onschadelijk,
maar DWD's eigen routine (`nlr_datasourcespecific.py`) gooit expliciet een
`Exception: Product not available` zodra het gevraagde "product" ('g', wat geen echt
DWD-bestandsproduct is) niet op schijf gevonden wordt. Die crash brak de hele
per-paneel-verwerking af vóórdat de eigenlijke 'g'-specifieke logica (die wél correct Z/CC/
ZDR ophaalt, via `get_data_multiple_scans`) ooit werd bereikt — vandaar de melding "(OLD)",
hetzelfde signaal als bij echt ontbrekende bestanden.

## Fix
Deze normale, bron-specifieke aanroep wordt nu simpelweg overgeslagen voor product 'g' —
die was er toch nooit voor nodig, want `_calculate_polrgb` haalt Z/CC/ZDR zelf al apart en
correct op.

## Verificatie
Nagerekend dat de paar regels code die ná deze aanroep nog binnen dezelfde proberen-blok
staan (een live-data-controle, en een dealiasing-stap) voor product 'g' sowieso niet van
toepassing zijn of geen probleem geven als ze worden overgeslagen — geen ander gedrag is
aangepast.

---

**Update 14**: legenda iets verder van de paneelrand af gepositioneerd (meer ruimte naar
links en naar onder), naar aanleiding van feedback dat de balkjes te dicht tegen de hoek
en de oude linker-as-getallen aanstonden. Alleen `nlr_plotting.py` opnieuw aangepast.

## Wat er veranderde
De ene gedeelde `margin` (8px) is gesplitst in een aparte `margin_x` (40px) en `margin_y`
(28px), zodat de twee assen onafhankelijk afgesteld konden worden in plaats van met één
gedeelde waarde die op de ene as al genoeg, op de andere nog te weinig marge gaf.

---

**Update 13**: een dunne, witte rand om elke legendabalk, naar het voorbeeld van Bram's
screenshot. Alleen `nlr_plotting.py` opnieuw aangepast.

## Wat er veranderde
De balken hadden al een `border_width` ingesteld, maar geen expliciete `border_color` —
vispy's standaardwaarde daarvoor is zwart, wat op onze zwarte balkachtergrond uiteraard
onzichtbaar is. Nu een zachte witte rand (`border_color=(1,1,1,0.8)`, dus niet knalwit
maar net iets transparant) en de dikte iets verhoogd (1 → 1.5px) voor een duidelijker,
verzorgder randje.

---

**Update 12**: tekst-overlap in de legenda opgelost — de balkjes zelf stonden al goed
(zichtbaar in het screenshot na Update 11), maar de getallen en labels van de drie kolommen
vielen over elkaar heen. Alleen `nlr_plotting.py` opnieuw aangepast.

## Oorzaak
De balken stonden met maar 10px breedte en 3px tussenruimte naast elkaar (~13px tussen de
centra) — veel te weinig voor leesbare tick-tekst zoals "100.0", die al 25-30px breed is.
Daarnaast stond het label (incl. eenheid, bijv. "ρHV (%)") onder de balk, op dezelfde hoogte
als de tick-tekst van de balk ernaast, wat een tweede overlap-bron was.

## Fix
- Balkbreedte en tussenruimte vergroot (centra nu ~70px uit elkaar in plaats van ~13px).
- Het label (met eenheid) staat nu boven de balk in plaats van onder, waar het niet meer met
  de tick-tekst van de naburige balk concurreert om ruimte.
- `bar_height_frac` iets verkleind (0.32 → 0.28) om ruimte te maken voor het label erboven
  binnen dezelfde totale legenda-hoogte.

Nog steeds niet zelf te verifiëren in mijn omgeving (geen draaiende GUI): of de exacte
afstanden nu ruim genoeg zijn op jouw scherm/resolutie, en of de kleurverloop-richting in de
balken klopt (zwart onder, volle kleur boven). Beide kun je nu in elk geval zien, in
tegenstelling tot de vorige twee updates.

---

**Update 11**: twee fundamentele fouten in de Update 10-legenda gevonden en gefixt aan de
hand van de console-output en het screenshot. Alleen `nlr_plotting.py` opnieuw aangepast.

## Fout 1 (crash): `KeyError: 'polrgb_legend_ticks'` in `on_resize`
`self.font_sizes` (het dict dat per tekst-visual aangeeft welke instelling zijn lettergrootte
bepaalt) was niet uitgebreid met entries voor de twee nieuwe tekst-visuals
(`polrgb_legend_ticks`, `polrgb_legend_labels`). Bij elke resize (ook de allereerste, bij het
opstarten) crashte `on_resize` daardoor, wat op zijn beurt een vervolgfout gaf
(`TypeError` in vispy's tekst-rendering, omdat de DPI-transform door de crash nooit correct
was geïnitialiseerd) en uiteindelijk het hele scherm liet vastlopen in een onbruikbare staat
(zoals te zien op het gedeelde screenshot: een groot zwart vlak, vervormde lay-out, géén
legenda zichtbaar).

Fix: twee regels toegevoegd aan `self.font_sizes`, die hergebruiken dezelfde instelling als
de bestaande `cbars_ticks`/`cbars_labels`.

## Fout 2 (geen crash, maar fundamenteel verkeerde aanpak): balkjes zouden meebewegen met
## pan/zoom
Ik had de drie legenda-balkjes per paneel opgeslagen als een dict binnen `visuals_panels`
(de lijst van visual-soorten die normaal bij het *radarbeeld zelf* horen). Dat is een
categorie die automatisch de paneel-specifieke pan/zoom-transform en clipper krijgt
toegewezen (zie de transform-toewijzing in `set_panel_sttransforms_and_clippers`) — prima
voor de kaart en de radardata zelf, maar niet voor een vast-in-de-hoek-gepositioneerde
legenda-overlay. Dit zou de balkjes (als ze al zichtbaar waren geweest, zie fout 1) hebben
laten meebewegen/vervormen bij het in- of uitzoomen, in plaats van een vaste positie en
grootte te behouden.

Fix: de balkjes zijn geherstructureerd als platte, individueel benoemde visuals
(`polrgb_legend_bar`+paneelnummer+kanaal, bijv. `polrgb_legend_bar0r`), naar het patroon van
de al langer bestaande `cbar0`..`cbar9`-visuals, die wél in `visuals_global` zitten en dus
geen pan/zoom-transform krijgen. `set_polrgb_legend` is overeenkomstig bijgewerkt.

## Nog te verifiëren (kon ik nog steeds niet zelf zien renderen)
- De richting van het kleurverloop in elke staaf (zwart onder, volle kleur boven — zie de
  notitie hierover in de code, met de triviale fix als het omgekeerd blijkt).
- De exacte positie/grootte/marges nu de coördinaten-aanpak (downward-y, `panel_corners`)
  voor het eerst zonder de eerdere font_size-crash kan worden getoond.

---

**Update 10**: kleine Bram-stijl legenda (3 verticale staafjes: Z=rood, CC=groen, ZDR=blauw,
met hun min/max-waarden) toegevoegd in de hoek van elk paneel dat product 'g' toont. Alleen
`nlr_plotting.py` aangepast.

## Let op: dit kon ik niet visueel testen
Dit raakt vispy's lage-niveau positionerings- en shader-gedrag (canvas-pixelcoördinaten,
tekst-ankers, colorbar-richting), en ik heb geen draaiende GUI in mijn omgeving om dit zelf
te zien renderen. Ik heb de logica zo zorgvuldig mogelijk afgeleid door de bestaande,
bewezen werkende cbar-code (`set_individual_cbar`) regel voor regel te volgen, maar twee
dingen kun je het beste eerst zelf checken voordat je erop vertrouwt:

1. **Richting van de kleurgradient binnen elke staaf.** Bedoeld: zwart onderaan, volle
   kleur (rood/groen/blauw) bovenaan — zoals in Bram's screenshot. Als het bij jou
   omgekeerd blijkt, is de fix triviaal: in `Plotting.__init__` de regel
   `cmap = color.Colormap([(0,0,0,1), channel_color])` aanpassen naar
   `color.Colormap([channel_color, (0,0,0,1)])` (de twee kleuren omdraaien).
2. **Exacte positie/grootte in de hoek.** Ik gebruik dezelfde downward-y schermcoördinaten
   (`panel_corners`) als de rest van het ongetransformeerde teken-systeem, maar dit is de
   eerste keer dat ik zelf nieuwe visuals op die manier positioneer in plaats van bestaande
   code aan te passen. Als de balkjes op een vreemde plek staan of overlappen met de
   radarkaart, ligt dat waarschijnlijk aan een teken- of schaalfout in `set_polrgb_legend`.

## Wat er is toegevoegd

### Nieuwe visuals (in `Plotting.__init__`)
- Drie `ColorBarVisual`-objecten per paneel (`polrgb_legend_bar_r/g/b`), elk met een simpele
  2-punts colormap (zwart → volle kanaalkleur), ticks en eigen label uitgeschakeld (we tekenen
  die zelf, net als bij de bestaande cbars).
- Twee gedeelde `TextVisual`-objecten (`polrgb_legend_ticks`, `polrgb_legend_labels`) voor de
  getallen en de Z/CC/ZDR-labels, naar het patroon van de bestaande `cbars_ticks`/`cbars_labels`.
- Geregistreerd in `visuals_order`, `visuals_panels` en `visuals_widgets['main']`, op dezelfde
  manier als de bestaande cbar-visuals.

### Nieuwe functie: `set_polrgb_legend`
- Toont de drie staafjes alleen voor panelen die nu daadwerkelijk product 'g' tonen; verbergt
  ze voor alle andere panelen/producten.
- Positioneert ze in de linkerboven-hoek van het paneel (marge instelbaar via de constanten
  `margin`/`bar_width`/`gap`/`bar_height_frac` boven in de functie).
- Leest de min/max-waarden direct uit `self.gui.polrgb_params` (dezelfde dict als de Settings
  → PolRGB-tab), dus de legenda toont altijd de actuele instelling.
- Wordt aangeroepen vanuit `set_newdata` (na `set_cbars`), dus bij elke redraw, productwissel
  of parameterwijziging in Settings (via de bestaande `set_newdata`-aanroep die
  `change_polrgb_param`/`reset_polrgb_params` al deden) wordt de legenda vanzelf bijgewerkt.

## Geen wijzigingen nodig in nlr.py of nlr_datasourcegeneral.py
Deze update hergebruikt alleen `self.gui.polrgb_params` (al aanwezig sinds Update 7) en de
bestaande `set_newdata`-aanroepen vanuit de Settings-tab (al aanwezig sinds Update 8) — er
was geen nieuwe haak nodig in die bestanden.

---

**Update 9**: cursor-uitlezing (de statusbalktekst die meebeweegt met de muis) toont nu voor
product 'g' de losse Z/CC/ZDR-waarden onder de cursor, in plaats van enkel '--'. Bestanden
gewijzigd: `nlr_plotting.py` en `nlr_datasourcegeneral.py`.

## Wat er is toegevoegd

### nlr_datasourcegeneral.py
`_calculate_polrgb` cachet nu de rauwe (niet-genormaliseerde, fysieke) Z/CC/ZDR-arrays per
paneel in een nieuw dict `self.polrgb_raw_data` (geïnitialiseerd in `__init__`), vlak voordat
ze worden omgezet naar kleurkanalen. Dit gebeurt sowieso al binnen de functie, dus dit kost
geen extra ophalingen — het is puur het bewaren van iets dat al berekend werd.

### nlr_plotting.py
`update_data_readout` (de functie achter de statusbalk-tekst) had voorheen voor product 'g'
een `raise Exception(...)` die altijd naar '--' viel terug, met de redenering dat een
RGB-pixel geen eenduidige scalar-waarde heeft. Dat klopt voor de weergegeven kleur zelf, maar
niet voor de onderliggende meetwaarden. Nu leest de functie voor 'g' de drie waarden uit
`self.dsg.polrgb_raw_data[panel]` op de pixel onder de cursor (dezelfde `row`/`col`-bepaling
die al generiek voor alle producten gebeurt) en toont een tekst in de vorm:
```
Z=35.9 dBZ, CC=96.8 %, ZDR=0.2 dB
```
in plaats van de normale 'waarde/min-in-beeld/max-in-beeld'-opbouw die voor scalar-producten
gebruikt wordt (die opbouw slaat nergens op bij drie losse grootheden, dus is voor 'g'
overgeslagen).

Als de cache voor dat paneel om wat voor reden dan ook ontbreekt (bijv. een mislukte
Z/CC/ZDR-ophaling), valt de uitlezing terug op '--' zoals voorheen, zonder crash.

## Waarom dit handig is
Tijdens de Vroomshoop-analyse moest je nog handmatig wisselen naar de losse 'c'- en
'd'-producten om de exacte CC/ZDR-waarde op een interessante plek te zien. Met deze update
kun je dat nu direct aflezen terwijl je naar de PolRGB-composite zelf kijkt, zonder te
wisselen van product.

---

**Update 8**: crash-fix bij het openen van Settings (`AttributeError: 'GUI' object has no
attribute 'polrgb_params'`). Alleen `nlr.py` opnieuw aangepast.

## Oorzaak
In Update 7 werd `polrgb_params` als module-level variabele gedefinieerd en netjes
opgenomen in de instellingen-persistentie (laden/opslaan via pickle), maar ik vergat de
regel die deze module-level waarde ook daadwerkelijk naar het GUI-object zelf kopieert
(`self.polrgb_params = polrgb_params`) — exact het patroon dat ook voor `cmaps_minvalues`
en `cmaps_maxvalues` wordt gebruikt, een paar regels erboven in dezelfde `__init__`. Zonder
die regel bestond `self.polrgb_params` simpelweg niet, vandaar de crash zodra de nieuwe
Settings-tab geopend werd en daar `self.polrgb_params[key]` opvroeg.

## Fix
Eén regel toegevoegd in `GUI.__init__`, direct naast de bestaande
`self.cmaps_minvalues=cmaps_minvalues` / `self.cmaps_maxvalues=cmaps_maxvalues`:
```
self.polrgb_params=polrgb_params
```

---

**Update 7**: alle 11 PolRGB-parameters nu instelbaar via de GUI (Settings -> PolRGB), in
plaats van hardcoded in de code. Bestanden gewijzigd: `nlr.py` (nieuw in deze set) en
`nlr_datasourcegeneral.py`.

## Waarom
We hebben in dit gesprek meerdere keren een parameter aangepast (Z-fade-grenzen, gamma,
ZDR-schaal, en weer terug) en daarvoor steeds een nieuw codebestand moeten aanleveren. Met
deze update kun je dat zelf direct in de GUI doen, zonder bestanden te vervangen of NLradar
opnieuw te starten.

## Wat er is toegevoegd

### nlr.py
- Module-level `polrgb_params_default` (dict met de 11 huidige standaardwaarden) en
  `polrgb_params` (de actief gebruikte waarden, een kopie van de defaults).
- `polrgb_params` toegevoegd aan de bestaande instellingen-persistentie (`variables_names_raw`
  / `variables_names_withclassreference`), dus het wordt net als andere instellingen
  opgeslagen in `Generated_files/stored_settings.pkl` en bij de volgende start automatisch
  weer geladen.
- Een defensieve aanvulling bij het laden: als een oudere opgeslagen `polrgb_params` een
  sleutel mist (bijvoorbeeld na een toekomstige uitbreiding met een 12e parameter), wordt die
  aangevuld met de huidige default in plaats van later een KeyError te geven.
- Nieuwe Settings-tab **'PolRGB'** (naast 'Color tables'), met voor elk van de 11 parameters
  een tekstveld plus duidelijk label, en een 'Reset to defaults'-knop. Wijzigingen worden
  direct toegepast (zelfde patroon als de bestaande colortable-instellingen) en het beeld
  wordt automatisch herberekend voor alle panels die op product 'g' staan.
- De 11 instelbare parameters: `Z_MIN`, `Z_MAX`, `CC_MIN`, `CC_MAX`, `ZDR_MIN`, `ZDR_MAX`,
  `Z_FADE_LO`, `Z_FADE_HI`, `ALPHA_GAMMA`, `CC_FALLBACK`, `ZDR_FALLBACK` — exact dezelfde die
  voorheen hardcoded in `_calculate_polrgb` stonden.

### nlr_datasourcegeneral.py
- `_calculate_polrgb` leest de 11 waarden nu uit `self.gui.polrgb_params` in plaats van ze
  als hardcoded constanten te definiëren. Elke aanroep leest opnieuw uit die dict, dus een
  wijziging in Settings werkt door zonder herstart.
- Een hardcoded `fallback_defaults`-dict binnen de functie zelf vangt af als er ooit een
  sleutel in `self.gui.polrgb_params` zou ontbreken (zou niet moeten gebeuren gezien de
  aanvulling in nlr.py, maar dit voorkomt een crash in plaats van enkel te vertrouwen op die
  ene plek).

## Belangrijke kanttekening over caching
De berekende RGBA-data voor product 'g' wordt, net als andere producten, in het geheugen
gecached zodat niet bij elke redraw opnieuw Z/CC/ZDR van schijf gelezen worden. Een aanpassing
van een PolRGB-parameter maakt die cache nu expliciet ongeldig voor product 'g' (via
`self.pb.cmap_lastmodification_time['g']`, hetzelfde mechanisme dat al bestond voor
colormap-wijzigingen bij andere producten) — zonder die stap zou een parameterwijziging in de
Settings-tab pas zichtbaar worden bij de volgende keer dat de scan/tijd toch al wijzigt.

---

**Update 6**: ZDR-schaal terug naar 0..3 dB. Alleen `nlr_datasourcegeneral.py` opnieuw
aangepast.

## Waarom
Update 5 verbreedde de ZDR-schaal naar -8..3 dB om negatieve ZDR (hagelindicator) zichtbaar
te maken in plaats van geklemd op 0. Na een live vergelijking op dezelfde Mander-case bleek
het neveneffect (gewone regen oogt cyaan-getint i.p.v. verzadigd groen) in de praktijk
storender dan de winst van het zichtbaar maken van negatieve ZDR waard was. Op uitdrukkelijk
verzoek terugverzet naar de oorspronkelijke schaal.

## Wat er verandert
`ZDR_MIN, ZDR_MAX` terug van `-8.0, 3.0` naar `0.0, 3.0`. Verder niets gewijzigd — de
RGBA-transparantie, de Z-fade/gamma-instelling, en de CC/ZDR-fallbacks blijven zoals in
Update 5.

## Consequentie om te onthouden
Met deze schaal wordt sterk negatieve ZDR (een belangrijke hagelindicator, zoals de -7.4 dB
die bij Vroomshoop werd gevonden) weer naar blauw=0 geklemd, en is dus niet meer rechtstreeks
in de RGB-kleur te zien. Voor een hageldiagnose blijft het raadzaam om de losse ZDR-plot
(toets 'd') te checken in plaats van op de RGB-kleur te vertrouwen — zie ook de PolRGB
kleurgids, sectie 4, stap 3-4.

**Let op:** de PolRGB-kleurgids (PolRGB_kleurgids.docx / .md) is bijgewerkt naar de -8..3 dB
schaal in een eerdere stap van dit gesprek. Met deze terugdraai klopt die gids niet meer
met de huidige code — die moet opnieuw worden teruggezet naar de 0..3 dB beschrijving als
deze nog gebruikt wordt.

---

**Update 5**: ZDR-schaal verbreed van 0..3 dB naar -8..3 dB. Alleen `nlr_datasourcegeneral.py`
opnieuw aangepast.

## Waarom
Sterk negatieve ZDR (tuimelende/niet-afgeplatte hagelstenen) werd voorheen altijd naar 0 dB
geklemd, dus optisch niet te onderscheiden van "ZDR is ongeveer 0". Bij een echte case
(Vroomshoop, 2 sept 2024) bleek de losse ZDR-plot een waarde van -7.4 dB te tonen op een
plek die in de RGB gewoon als "neutraal blauw=0" oogde — een gemiste kans om een
hagelsignaal direct in de RGB te laten zien zonder naar de losse plot te moeten schakelen.

## Wat er verandert
`ZDR_MIN, ZDR_MAX` van `0.0, 3.0` naar `-8.0, 3.0`. `ZDR_FALLBACK` blijft op 0.5 dB
(een fysieke aanname over typische lichte regen, geen positie-op-de-schaal-keuze, dus die
hoeft niet aangepast te worden alleen omdat de schaal breder werd).

**Bewust geaccepteerd neveneffect**: omdat het ZDR-bereik van gewone regen (~0-2 dB) nu een
kleiner deel van de totale schaal beslaat, oogt gewone regen niet meer als zuiver verzadigd
groen maar als een lichtere, cyaan-getinte groene kleur. Ook de bekende clutter-ruis dicht
bij de radar wordt door deze verbreding sterker zichtbaar (meer blauw/paars), omdat clutter
vaak ook chaotische/negatieve ZDR-achtige ruis vertoont die nu niet meer wegvalt tegen 0.
Beide effecten zijn getest en met Erik besproken vóór implementatie; de voorkeur ging uit
naar het zichtbaar maken van het hageldiagnostische signaal, ook al verandert daarmee het
uiterlijk van gewone regen en clutter.

---

**Update 4**: zwarte achtergrond binnen de radarcirkel opgelost — de kaart (satellietbeeld/
grenzen) schemerde niet door op plekken zonder neerslag. Alleen `nlr_datasourcegeneral.py`
opnieuw aangepast.

## Fix: RGBA in plaats van RGB
We gaven tot nu toe een 3-kanaals `(azimuth, range, 3)` array door, waarbij we zelf al
(onzichtbare) pixels vooraf naar zwart mengden. vispy's `ImageVisual` beschouwt 3-kanaals
data echter als **volledig opaak** — er is geen transparantie-informatie, dus de kaart
eronder (die wel degelijk eerder wordt getekend, zie `visuals_order`) werd overschilderd
door ons zwart in plaats van te kunnen doorschijnen.

De radar-visuals zijn al ingesteld met de `'translucent'` GL-state (`blend=True,
blend_func=(src_alpha, one_minus_src_alpha)`) — dezelfde stand die ook bij normale,
colormap-gebaseerde producten voor transparantie zorgt. Die had alleen een echt
alpha-kanaal nodig om iets te kunnen doen.

`_calculate_polrgb` geeft nu een `(azimuth, range, 4)` RGBA-array terug: R/G/B zoals
voorheen, en als 4e kanaal onze al-berekende zichtbaarheids-`alpha` (dezelfde die eerder
gebruikt werd om met een hardcoded zwarte achtergrond te mengen — nu laten we de GPU zelf
mengen met wat er werkelijk onder ligt). Ook de foutafhandeling (als het ophalen van
Z/CC/ZDR faalt) geeft nu een volledig transparante RGBA-array terug in plaats van een
opaak zwart vlak.

Geen wijzigingen nodig in `nlr_plotting.py`: vispy detecteert zelf, puur op basis van het
aantal kanalen in de array (`shape[2] == 3` vs. `4`), of het om RGB of RGBA gaat, en past
de blending overeenkomstig toe.

---

**Update 3**: zwarte gaten in overigens duidelijke neerslag opgelost (geen crash, geen
verkeerde rendering — een afweging in de kleurmapping zelf). Alleen `nlr_datasourcegeneral.py`
opnieuw aangepast.

## Analyse van de zwarte gaten (n.a.v. screenshot met grote bereik-scan, r~400km)
Onderzocht met een tweede testbestand (RAD_NL62_VOL_NA_202507021505.h5, dezelfde 0.3°-tilt
maar het lange-bereik exemplaar, ~320km i.p.v. ~30km). Twee deeloorzaken gevonden:

1. **CC/ZDR missen vaker dan Z bij zwak signaal** (fysisch normaal: polarimetrische schattingen
   hebben een hogere signaal-ruisverhouding nodig dan een ruwe Z-meting). Bij ~8% van de
   pixels met geldige Z ontbreekt CC of ZDR. Eerste fix: CC/ZDR vallen nu **onafhankelijk**
   terug op een neutrale 'typische lichte regen'-waarde (CC_FALLBACK=97%, ZDR_FALLBACK=0.5dB)
   wanneer ze missen maar Z wel geldig is, in plaats van de hele pixel te blanken.
2. **Het grotere effect**: een breed gebied met fysisch zwak signaal (Z tussen -10 en 0 dBZ,
   middenin een verder duidelijk regengebied — geen databug, gewoon een lokale dip in
   intensiteit) viel grotendeels binnen onze Z-fade (was -5..+10 dBZ) en werd daardoor
   (bijna) volledig zwart/transparant, wat in het beeld als een hard, onnatuurlijk gat oogt.

   Clutter dichtbij de radar en zwak-maar-reëel signaal verder weg overlappen elkaar te veel
   in zowel Z als CC om met een eenvoudige drempel te onderscheiden (geprobeerd, werkte niet
   afdoende — zie analyse in eerdere iteraties als je de tussenstappen wil zien). Met Erik
   afgestemd: bewuste keuze voor **iets meer clutter dichtbij de radar accepteren**, in ruil
   voor minder/zachtere zwarte gaten in zwakke neerslag verder weg.

## Fix: bredere fade + gamma-correctie (nlr_datasourcegeneral.py, `_calculate_polrgb`)
- `Z_FADE_LO, Z_FADE_HI` van `-5, 10` naar `-15, 10` — laat zwakker signaal eerder meedoen.
- Nieuwe `ALPHA_GAMMA = 0.6` op de alpha-curve (vóór de eindpunten 0/1, alleen het
  tussenliggende verloop wordt opgetrokken) — geeft zwak-maar-net-boven-de-afsnijgrens
  liggend signaal een hogere zichtbaarheid dan een lineaire fade zou doen, zonder de
  clutter-onderdrukking bij de allerlaagste Z-waarden te verliezen.
- Beide waarden staan als losse, duidelijk benoemde constanten boven in de functie — voel je
  vrij om ze verder te finetunen terwijl je live door verschillende momenten/radars bladert;
  dat gaat sneller dan dat ik blind op losse testbestanden verder optimaliseer.

Resultaat op het testbestand: aandeel (bijna-)zwarte pixels in het bewuste gat-gebied daalde
van 47% naar 49%... eigenlijk: het zwarte-pixel-aandeel verschilt weinig in getal, maar de
verdeling verschuift van 'hard zwart' naar 'donker maar zichtbaar groen', wat in de praktijk
veel natuurlijker oogt (zie ook de gerenderde voorbeelden tijdens het ontwikkelproces).

---

**Update 2**: derde fix, dit keer geen crash maar een verkeerd renderresultaat — het beeld
toonde een doorlopende Z-achtige regenboogkleur in plaats van de groen/wit/roze RGB-mapping.
Oorzaak gevonden en opgelost in `nlr_plotting.py`.

## Fix 3 — verkeerde kleuren (geen crash, wel fout beeld)
vispy's `ImageVisual` bepaalt intern, via een GLSL "color transform"-functie, of het de data
als scalar+colormap moet tonen of als kant-en-klare RGB. Die functie wordt echter alleen
herbouwd op het moment dat de `cmap`-**property** opnieuw wordt toegekend — niet automatisch
wanneer `set_data()` met een andersvormige array wordt aangeroepen.

Omdat we de `cmap`-toewijzing voor product 'g' bewust oversloegen (terecht, want een RGB-array
heeft geen colormap nodig), kreeg vispy nooit het signaal om de color-transform-functie
opnieuw op te bouwen. Het paneel bleef daardoor de shader gebruiken die was opgebouwd voor het
vorige (scalar) product op dat paneel — meestal de Z-colormap, vandaar het regenboog-effect.

Fix: in `set_newdata` wordt nu per paneel en per visual-type (`radar_polar`/`radar_cartesian`)
bijgehouden of de laatst opgebouwde color-transform RGB-passthrough of scalar+cmap was
(`self.visual_colortransform_is_rgb`, nieuw, geïnitialiseerd in `__init__`). Zodra een paneel
wisselt van/naar product 'g', wordt `_need_colortransform_update` op de visual handmatig op
`True` gezet, zodat vispy bij de volgende draw de juiste (RGB-passthrough) functie opbouwt.

Dit attribuut bestaat al in vispy's `ImageVisual.__init__` (vóór `freeze()` wordt aangeroepen),
dus het is een gewone attribuutwijziging, geen nieuwe attribuut-toevoeging aan een frozen
object — dat laatste zou een `AttributeError` hebben gegeven.

---

**Update 1**: twee crash-fixes na de eerste test-run met 'G'. Nu 4 bestanden gewijzigd
i.p.v. 3 — vervang ze allemaal in `Python_files/`.

## Crash-fixes (n.a.v. de traceback)

### 1. `nlr_importdata.py` — `calibration_C_formulas` niet gevonden
`get_data_multiple_scans` haalde de calibratieformule op met `'calibration_'+i_p.upper()+'_formulas'`.
Voor product `'c'` geeft dat `'calibration_C_formulas'`, terwijl het echte attribuut in het
HDF5-bestand `calibration_RhoHV_formulas` heet (en voor 'd' zonder directe ZDR-dataset:
`calibration_Zv_formulas`). Dit is een **al bestaande bug** in deze functie — hij viel tot nu
toe niet op omdat de functie voorheen alleen met product `'v'`/'z' werd aangeroepen (VWP, plain
products), waarbij `i_p.upper()` toevallig identiek is aan de echte attribuutnaam (`'V'`, `'Z'`).
Onze nieuwe `_calculate_polrgb`-aanroepen met `'c'` en `'d'` waren de eerste die dit blootlegden.
Fix: gebruik `productname` (al correct bepaald op de regel erboven) i.p.v. `i_p.upper()`.
Geen regressie mogelijk voor bestaande functionaliteit, want niets anders riep deze functie
ooit aan met product 'c' of 'd'.

### 2. `nlr_plotting.py` — `ValueError: too many values to unpack (expected 2)`
Drie plekken deden `azimuthal_bins, radial_bins = self.dsg.data[j].shape` (of soortgelijk),
wat een 2D-array verwacht. Onze RGB-data heeft shape `(azimuth, range, 3)` — 3 waarden i.p.v. 2.
Fix: `.shape[:2]` i.p.v. `.shape` op de drie betreffende regels (in `set_newdata` en `on_draw`).

Daarnaast: de cursor-uitleesfunctie (`update_data_readout`, toont de datawaarde onder de muis
in de statusbalk) ging er vanuit dat `self.dsg.data[panel][row,col]` een scalar geeft. Voor
'g' geeft dat een 3-elements RGB-tripel, wat geen bruikbaar getal voor de statusbalk oplevert.
Dit gaf geen crash maar wel een onzinnige uitlezing; nu toont de statusbalk gewoon '--' voor 'g'.

---

## nlr_globalvars.py
Product `'g'` geregistreerd op alle plekken waar producten worden opgesomd:
- `products_all`, `i_p['g']='z'` (scan-attributen volgen Z)
- `productnames`, `productnames_cmaps`, `productnames_cmapstab`, `productunits_default`
- `products_data_nbits`, `products_maxrange`, `cmaps_maxrange` — placeholder-waarden (0-255),
  niet functioneel gebruikt (RGB-data gaat niet door de scalar→uint pijplijn), maar wel nodig
  omdat meerdere plekken in nlr.py/nlr_background.py generiek over `products_all` itereren.
- `products_with_tilts` — 'g' toegevoegd zodat het normale single-tilt gedrag (scankoppeling
  tussen panels, scan-navigatie) werkt.
- `colortables_dirs_filenames_Default['g']` — placeholder (hergebruikt colortable_Z.csv),
  nooit echt gebruikt voor het tekenen, alleen om een KeyError in de Settings-GUI te voorkomen.

Sneltoets: `G` (automatisch via de bestaande generieke `for product in gv.products_all` loop
die shortcuts registreert in nlr.py).

## nlr_datasourcegeneral.py
`_calculate_polrgb(self, j)` herschreven (was aanwezig maar nooit aangesloten/niet werkend).
Wordt aangeroepen vanuit `get_data()` zodra `self.crd.products[j] == 'g'`.

Werking:
1. Haalt Z, CC (RhoHV) en ZDR op voor de huidige scan via
   `self.source_classes[self.data_source()].get_data_multiple_scans(...)` — dezelfde
   generieke interface die ook VWP en de plain-products gebruiken. Voor KNMI berekent dit
   pad ZDR automatisch uit Zh/Zv met gecombineerde maskering wanneer er geen direct
   ZDR-dataset in het bestand staat (is het geval voor de nieuwe KNMI-radars).
2. Combineert de maskers (NaN bij missende data in Z, CC, of ZDR) tot één `nodata`-masker.
3. Normaliseert: R=Z (-10..60 dBZ), G=CC (70..100%), B=ZDR (0..3 dB) — Bram's oorspronkelijke
   ranges.
4. Past een vloeiende alpha-fade toe op basis van Z (-5..+10 dBZ): onder -5 dBZ volledig
   transparant/zwart, erboven geleidelijk zichtbaar. Dit voorkomt dat laag-Z/laag-CC clutter
   dicht bij de radar (waar de Zh-Zv aftrekking voor ZDR ruisgevoelig is) als gekleurde
   speckle verschijnt.
5. Geeft een `(azimuth, range, 3)` uint8-array terug.

## nlr_plotting.py
- `cmap`/`clim`-toewijzing aan de vispy `ImageVisual` wordt overgeslagen voor product `'g'`
  (de RGB-array heeft geen colormap nodig).
- Drie `.shape`-unpackings aangepast naar `.shape[:2]` (zie crash-fix 2 hierboven).
- Cursor data-readout (statusbalk) toont '--' voor product 'g' i.p.v. een onzinnige waarde.

## nlr_importdata.py
- Bugfix in `get_data_multiple_scans` (zie crash-fix 1 hierboven): `productname` i.p.v.
  `i_p.upper()` bij het opzoeken van de calibratieformule.

## Bekende beperkingen / nog open
- De colorbar naast een 'g'-paneel toont nog steeds een (betekenisloze) Z-achtige schaal.
  Functioneel onschadelijk, kosmetisch nog niet opgeruimd — bewust uitgesteld.
- `get_data_multiple_scans` binnen `_calculate_polrgb` gaat niet via de centrale
  in-memory cache (`self.stored_data`); bij elke herberekening van 'g' worden Z/CC/ZDR
  opnieuw van schijf gelezen. Functioneel correct, maar niet maximaal snel.
- Alleen voor KNMI geïmplementeerd en getest.
- `productunfiltered`/`polarization` worden voor 'g' hardcoded op `False`/`'H'` gehouden.

## Geteste configuratie
- Kleurranges: Z -10..60 dBZ, CC 70..100%, ZDR 0..3 dB (Bram's oorspronkelijke keuze)
- Getest tegen scan7 (0.3° tilt) van RAD_NL62_VOL_NA_202507021540.h5 — toont duidelijk
  groene neerslagvelden met witte/roze convectieve kernen, vergelijkbaar met het
  ARPA Lombardia-voorbeeld.
- De twee crash-fixes zijn alleen gevalideerd via syntax-check + geïsoleerde logica-test
  (geen PyQt5/h5py/GPU beschikbaar in mijn omgeving om de volledige GUI te draaien) — dus
  graag opnieuw testen met 'G' en de eventuele nieuwe traceback terugkoppelen als het nog
  niet helemaal goed gaat.
