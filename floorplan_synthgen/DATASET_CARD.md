# Synthetic Floor-Plan Dataset — Implementation, Data & Rendering Spec

A complete description of **how the dataset is produced**, **what each artifact
contains**, the **statistical characteristics** of the data, and **how plans are
(and should be) rendered**. This is the reference for anyone consuming or
re-rendering the dataset.

---

## 1. What this is

A procedural generator that emits architectural floor plans for training a
wall/opening-extraction model. Each plan is produced as four aligned artifacts:

| artifact | what it is |
|---|---|
| `configs/plan_XXXXX.json` | full vector geometry of the plan (walls, rooms, openings) in **millimetres** |
| `images/<split>/plan_XXXXX.png` | the rendered architectural drawing (raster) |
| `rich_json/plan_XXXXX_rich.json` | per-opening records with **mm↔pixel** coordinates (the alignment bridge) |
| `labels/<split>/plan_XXXXX.txt` | YOLO detection labels (door/window boxes) |

Every emitted plan is **guaranteed valid** by a strict topological + semantic
validator (see §6). The schema is stable — additive only.

---

## 2. Pipeline (implementation)

Three stages, each a standalone, resumable script:

```
                config_generator.py                 render_dataset.py
  (seed,index) ───────────────────▶ config JSON ───────────────────▶ PNG + rich_json + YOLO
                  │  per-index, deterministic            │  ezdxf → matplotlib (Agg)
                  ▼                                       ▼
            validate_plan.py  (strict gate)        ModelTransform  (mm↔px)
```

1. **`config_generator.py` — procedural engine (CPU-cheap).**
   For each integer `index`, a per-index RNG fixes the *structural identity*
   (building type, footprint shape, size, room count, curved-wall bucket,
   region). It then BSP-partitions the footprint into rooms, builds walls from
   the room edges, places doors on a spanning tree of the room-adjacency graph
   (so every room is reachable), adds windows/entrances, columns, furniture and
   clutter. The candidate is passed through the strict validator; on failure the
   *interior layout* is re-drawn for the same identity (≤20 attempts) — so
   distributions stay on target regardless of per-attempt validity. Output:
   `plan_XXXXX.json` + `render_batches/batch_*.json` (sharded arrays for stage 2).

2. **`validate_plan.py` — strict validator (the contract).**
   `validate_plan(config) -> (is_valid, violations)` using `shapely` + `networkx`.
   Used both as the generation gate and as a standalone CLI/auditor. §6 lists the
   constraints.

3. **`render_dataset.py` — rasteriser.**
   Reads the config shards, rebuilds each plan as a DXF, renders to PNG via
   `ezdxf`'s matplotlib backend, and writes the YOLO label + `rich_json`. The PNG
   is published *after* its labels (a PNG on disk always implies its labels
   exist), making the stage **resumable**.

`generate_dataset.py` runs both stages with one command;
`generate_valid.py` is an all-in-one small-batch variant that also emits a
contact sheet.

---

## 3. Output layout

```
dataset/
  images/train/plan_XXXXX.png      labels/train/plan_XXXXX.txt
  images/val/plan_XXXXX.png        labels/val/plan_XXXXX.txt
  rich_json/plan_XXXXX_rich.json   # flat (not split)
  configs/plan_XXXXX.json          # flat
  data.yaml                        # Ultralytics descriptor
  render_report.json
```

Train/val split is deterministic per `plan_id` (MD5, ~15% val).

---

## 4. Schemas

### 4.1 `configs/plan_XXXXX.json` (units: **mm**)

Top level:
`id, group, name, description, hard_case, units("mm"), origin([0,0]), bbox([x0,y0,x1,y1]),
footprint({w,h}), counts{...}, walls[], rooms[], openings[], decoys[], clutter, render{}, metadata{}`

**`walls[]`** — every wall carries *both* a centerline+thickness *and* a polygon band:

| field | type | meaning |
|---|---|---|
| `id` | `"W1"…` | wall id |
| `type` | `"interior"\|"exterior"` | partition vs. footprint-bounding |
| `thickness_mm` | int | band width (≈80–400, thin partitions to 50) |
| `length_mm` | int | `== |centerline|` |
| `centerline` | `[[x,y],[x,y]]` | the wall axis (2 pts) |
| `polygon` | `[[x,y]…]` | the band = centerline offset ±thickness/2 (4 pts straight; traces the arc when curved) |
| `arc` | `null \| {center,radius,a0,a1}` | set **iff** the wall is curved |
| `material`, `hatch`, `angle_class` | str | e.g. `SOLID_BRICK`, `ANSI31`, `orthogonal\|angled\|curved` |

**`rooms[]`** — a filled region (no thickness):
`id, name, shape("rectangle"|"l_shape"|"t_or_u_shape"|"quad"|"polygon"|"curved"), polygon[[x,y]…], room_type`

**`openings[]`** — a segment on a wall (door/window/opening):

| field | meaning |
|---|---|
| `id` (`D*`/`W*`), `category`(`door\|window\|opening`), `subtype` | identity |
| `width_mm` | `== |p1−p2|` (standard catalogue size, see §7) |
| `p1,p2` | endpoints on the host wall; `center` = midpoint |
| `angle_deg`, `height_mm`, `hinge`, `swing`, `panels`, `sill_mm`, `head_mm`, `plane` | leaf/sill metadata |
| `symbol` | `{lines, arcs, polylines, dashed}` — the drawn door-swing / window mullions |

### 4.2 `rich_json/plan_XXXXX_rich.json` (the mm↔px bridge)

`plan_id, scenario, scale, rotation_deg, image_w, image_h, openings[]`

Each opening record (door example): `id, type, subtype, clear_opening_mm,
center_px, bbox_model([mm]), bbox_px, bbox_normalized([cx,cy,w,h] 0–1),
swing_direction, max_swing_angle_deg, hinge_point_{mm,px}, leaf_end_{mm,px}`.
Window records carry `p1_{mm,px}, p2_{mm,px}, angle_deg` instead of hinge/leaf.
**`*_px` are pixel coords in the matching PNG**; `bbox_normalized` is the YOLO box.

### 4.3 YOLO labels + `data.yaml`

`labels/.../plan_XXXXX.txt`: one row per opening — `class cx cy w h` (normalized).
`data.yaml`: `nc: 2`, `names: [door, window]` (class `0`=door, `1`=window).
Columns, furniture, clutter, text and the title block are **distractors only** —
they are deliberately *not* labelled (hard negatives).

---

## 5. Coordinate system, units & alignment

- **Units:** millimetres; integer coordinates snapped to a **50 mm grid** so shared
  edges/junctions coincide exactly. `origin = [0,0]`; `bbox` is computed from wall
  *polygons* (so it is slightly larger than the room union by ~½ exterior thickness).
- **Rotation:** a plan may be rotated (see §7); when rotated, all geometry in the
  config is already rotated (coordinates are no longer axis-aligned).
- **mm → px:** the renderer returns a `ModelTransform`; `transform.to_px(x_mm,y_mm)`
  maps config millimetres to PNG pixels. This is the *same* transform used to write
  `rich_json`, so config geometry, `rich_json` and the PNG are **pixel-aligned**
  (verified: `to_px(opening.center)` matches the rich record to <0.1 px, even at
  45° rotation). To overlay walls on a PNG, map `walls[].polygon` through `to_px`.

---

## 6. Validity guarantees (every emitted plan passes)

`validate_plan.py` enforces, with `shapely` (geometry) and `networkx` (connectivity):

- **Topological (T1–T8):** watertight single exterior loop; **no floating walls**
  (every wall end joins another wall, a T-junction, or the boundary); closed room
  loops (every room edge covered by a wall); gap-free / overlap-free partition; no
  degenerate or duplicate walls; clean junctions (no mid-span crossings); valid
  arcs (chord ↔ arc consistent); simple room polygons.
- **Semantic (S1–S10):** **every room has a door**; **connected plan** (all rooms
  reachable from an exterior entrance); doors lie on a wall within span connecting
  2 rooms or room+exterior; windows on walls, no door overlap; openings on a wall
  don't overlap and fit; min room area ≥2 m²; realistic wall thickness; realistic,
  **standard opening widths**; room typing & counts consistent; **no wall crosses
  an opening** (S10 — no doorway blocked by a transverse wall).
- **Consistency (C1–C5):** schema fields present; `centerline↔length↔polygon`
  consistent and `arc` iff curved; opening `p1/p2/center/width` consistent and on
  the host wall; `units=="mm"`; `bbox`/`footprint` coherent.

Audit any set: `python validate_plan.py <dir-of-configs> --json report.json`.

> Context: the *original* 10k dataset scored only **~7%** valid under this
> validator — dominated by injected floating "nib" walls (T2, 86%) and walls
> crossing openings (S10, 72%). Both are fixed by construction here.

---

## 7. Data characteristics (distributions)

Per-plan structural identity is drawn from fixed weighted distributions
(measured ranges from a 50-plan sample shown where useful):

**Building type** (weight %): residential_apartment 25, residential_house 15,
office_open_plan 10, office_cellular 10, hotel_floor 8, hospital_ward 5,
school_classroom_block 5, retail_unit 5, restaurant_cafe 5,
mixed_use_ground_floor 5, industrial_unit 4, sports_facility 3.

**Footprint shape** (curved-wall frequency bucket none 55 / one 25 / few 15 /
many 5): rectangle, L_shape, T_shape, U_shape, irregular_polygon,
rectangle_with_bay, rectangle_with_curved_end, fully_curved_facade,
organic_multi_arc. → **~40–45 % of plans contain ≥1 curved wall.**

**Rooms/plan:** building-type dependent, overall **2–28** (e.g. apartment 3–9,
hotel_floor 8–28, office_cellular 6–22). Sample mean ≈ 8–9 rooms, ≈13 walls,
≈9 doors, ≈12 windows per plan.

**Openings:**
- doors — subtypes `SINGLE_HINGED, DOUBLE_HINGED, SLIDING, POCKET, BIFOLD, FRENCH,
  GARAGE, REVOLVING, FOLDING_PARTITION`; **standard widths** (mm): interior
  600–1050 (e.g. 686/762/838 imperial, 700/800/900 metric), entrance up to 1200,
  patio/garage/folding 1600–5400. (Curved-glazing widths equal the true arc chord.)
- windows — `CASEMENT, SLIDING, FIXED, BAY, AWNING, LOUVRE, CORNER, CLERESTORY,
  SHOPFRONT`; standard widths 400–4200 mm.

**Wall materials** (→ hatch + thickness): CAVITY_BRICK, SOLID_BRICK,
REINFORCED_CONCRETE, CONCRETE_BLOCK, TIMBER_STUD, METAL_STUD, GLASS_PARTITION,
WET_WALL_BLOCK, RAMMED_EARTH, ICF_INSULATED, STRUCTURAL_RC_CORE. Each plan has
≥3 distinct thicknesses.

**Drawing style (sampled per plan):**
- **scale** 1:20 (5%) / 1:50 (25) / 1:100 (40) / 1:200 (25) / 1:500 (5)
- **standard** AIA 40 / ISO 25 / BS 15 / GB 12 / DIN 8
- **rotation** 0° (50%) / 1–5° (20) / 5–45° (20) / 45–89° (10) → ~50% rotated
- **clutter** none 15 / light 30 / medium 30 / heavy 25 (furniture appears at
  medium+; columns by building type)
- **region** us 40 / eu 35 / uk 15 / other 10 (drives imperial vs metric sizing)
- **DPI** uniform **80–220** (see §8 for the cost/size implication)

`metadata{}` records the realised choices per plan (building_type,
footprint_shape, region, standard, scale, clutter_level, curved_wall_count,
complexity, has_columns, has_furniture, imperial_sizing). `hard_case` flags
plans that are intentionally difficult (heavy clutter, many curves, etc.).

---

## 8. Rendering specification (how it is / should be rendered)

The renderer (`generator/renderer.py`) is **CPU-only** (`ezdxf` → matplotlib
**Agg**); a GPU does not accelerate it.

- **Backend / colour:** matplotlib Agg, white background, **monochrome** (all
  geometry ACI colour 7 → black). `ColorPolicy.BLACK`, `BackgroundPolicy.WHITE`.
- **Line weights** (`LineweightPolicy.ABSOLUTE`, in 1/100 mm), per CAD layer:
  exterior wall 0.50, interior wall 0.35, door 0.25, glazing 0.18, columns 0.35,
  wall hatch 0.09, dims 0.13, text/annotation 0.18, title block 0.25, furniture
  0.13, grid 0.09 (dashed). A per-plan global multiplier (`light 0.6 / standard 1.0
  / heavy 1.6`) is applied.
- **CAD layers:** components `A-WALL-FULL, A-WALL-INTR, A-DOOR, A-GLAZ, A-COLS`
  (never contain text — enforced); distractor layers `A-WALL-PATT (hatch),
  A-ANNO-DIMS, A-ANNO-TEXT, A-ANNO-TTLB, A-FURN, A-GRID, A-MISC`.
- **Framing / size:** the model bbox is padded by `3% + 50 mm`; pixels-per-mm is
  solved from the plan **scale** and **DPI**, then clamped so the long side fits a
  target canvas. Observed PNG sizes **800–2000 px** per side (square-ish).
- **Output:** RGB PNG (monochrome content). Optional Gaussian `noise_std`
  degradation (off by default).
- **`--max-dpi N`** clamps the sampled DPI at render time (e.g. 150). DPI mostly
  affects **file size**, not render time (the cost is line/hatch drawing, not
  pixels). `rich_json` always records the *actual* `image_w/image_h`, so capping
  DPI never breaks mm↔px alignment.

**Performance & storage (important):**
- **~3.5–4 s/plan/core** — dominated by drawing line + **hatch** entities (~0.6 s)
  + `savefig` (~0.3 s) + DXF build (~0.4 s). So **30k ≈ ~30 core-hours**; to finish
  in ~3 h you need ~10 cores in parallel (e.g. 3–4 parallel CPU workers/sessions).
  The renderer pins BLAS to 1 thread/worker so multiprocessing actually scales.
- **Size:** at native DPI ≈ **0.5 MB/PNG → ~16 GB for 30k**. Use `--max-dpi 150`
  (and optionally grayscale) to cut this substantially.

To re-render or overlay yourself, reuse the bridge:
`scenario_to_config(config) → FloorPlan → render() → (w, h, transform)`, then
`transform.to_px(x_mm, y_mm)` for any config point (see `generate_valid.py`).

---

## 9. Reproducing the dataset

```bash
pip install -r requirements.txt

# full dataset (resumable); --workers = cores, --max-dpi caps size
python generate_dataset.py --count 30000 --output dataset --workers 8 --max-dpi 150

# configs only (fast; render later or on-the-fly)
python generate_dataset.py --count 30000 --output dataset --skip-render

# shard across parallel runs/sessions (mergeable; distinct plan ids + batches)
python generate_dataset.py --count 7500 --start 0     --output shard0 --max-dpi 150
python generate_dataset.py --count 7500 --start 7500  --output shard1 --max-dpi 150
# … shard k: --start k*count

# audit
python validate_plan.py dataset/configs --json check.json   # expect 100% valid
```

Determinism: a given `(seed, index)` always yields the same plan, so runs and
shards are reproducible. See `README.md` for the Kaggle (background) workflow and
the 3-hour parallelisation plan.

---

## 10. Caveats & tips

- **Use CPU, not GPU** for rendering (matplotlib is CPU-bound).
- **Configs are cheap, rasterising is the cost** — if your trainer can render on
  the fly, ship configs and rasterise in the dataloader.
- Distractors (columns/furniture/clutter/text/title block) are intentional hard
  negatives and are **not** in the YOLO labels — don't treat them as missed boxes.
- The `polygon` and `centerline+thickness` wall representations are redundant by
  design; consume whichever your pipeline prefers (they're kept consistent by C2).
- Curved facades: rooms are bounded by the wall **chord**; the arc bulge is
  exterior decoration (not a room), which is why the room union is chord-watertight.
