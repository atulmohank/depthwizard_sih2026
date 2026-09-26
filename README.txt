DepthWizard spike (SIH26175) - Step 1
=====================================
Goal: one satellite/aerial image -> height map -> 3D flythrough.

SETUP (once)
  pip install -r requirements.txt
  (you already have CUDA torch from BusSense - check:  python -c "import torch;print(torch.cuda.is_available())")

GET A TEST IMAGE
  Easiest: Google Earth / Google Maps satellite, zoom into a dense Bengaluru area
  (buildings visible, straight top-down, NOT the tilted 3D view), screenshot, crop, save as city.jpg.
  Try 2-3 different areas: dense city, apartments + trees, flat open land.

RUN
  run.bat city.jpg
  -> out\city\viewer.html  (open in Chrome, needs internet for three.js)
  -> out\city\side_by_side.png  (image | height map - use in the deck)

DETREND (on by default) - removes the fake tilt/dome
  run.bat images\city_zoom.jpg                         -> default (ground mode)
  run.bat images\city_zoom.jpg --reuse --ground-size 0.2  -> re-tune in seconds, no model rerun
  --ground-size: bigger if large buildings get flattened, smaller if the dome remains (0.08-0.25)
  --detrend plane : only removes a straight tilt     --detrend off : raw model
  Viewer: "Raw model" / "Fixed" buttons = live before/after
  side_by_side.png now = image | raw | fixed   (deck slide)

IF BUILDINGS LOOK LIKE HOLES:  run.bat city.jpg --invert
BETTER QUALITY (slower):       run.bat city.jpg --model base
SMOOTHER / SHARPER 3D:         add --grid 256  or  --grid 480

WHAT TO LOOK FOR (report back)
  1. Are tall buildings raised above roads/ground?
  2. Where does it fail? (shadows, flat roofs, trees, water)
  3. Screenshot of the viewer + side_by_side.png
