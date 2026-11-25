# FIR Documents Extractor — Image Preprocessing Pipeline

A Jupyter Notebook (`app.ipynb`) that cleans scanned police First Information Report (FIR) case files page by page using **OpenCV + NumPy only**: no third-party AI models, no cloud APIs, no external OCR engines.

> **Current focus: image preprocessing.** Page classification (labelling each page as IIF-1…IIF-5, medical, etc.) is **paused** until preprocessing is finished. The previous classification cells were removed from the notebook and saved in [archive/page_classification_cells.py](archive/page_classification_cells.py) for reference. Nothing in that file runs.

---

## Project rules

1. **Processing lives in `image_processing.py`; review lives in `app.ipynb`.** The notebook imports the module and runs the same functions step by step with tables and before/after views. `device.py` detects the machine and picks settings (see *Hardware & device profiles*).
2. **Data stays local.** No cloud AI or APIs. Image preprocessing uses classical image processing only (OpenCV, NumPy). Local open-weight models (e.g. Qwen3-VL on the GB10, planned for extraction) are allowed.
3. **Orientation follows the content.**
   - A landscape page with landscape content (medical tables, X-ray slips) **stays landscape**.
   - Pages are never forced into portrait.
   - A page is turned by 90°/180° only when its *text* is clearly sideways or upside down.
   - Small-angle deskew (±10°) straightens tilt but never changes portrait/landscape.
4. **Every step can be checked and reviewed.**
   - Each step records per-page metrics.
   - A step is skipped when the page doesn't need it.
   - A page is flagged for human review when its result looks doubtful.
   - A review cell at the end filters and shows only the flagged pages.
5. **No hard-coded image sizes.**
   - Every size is set in millimetres in `CONFIG` and converted for each page using its DPI. DPI is estimated per document (median of its pages) or set with `dpi_override`.
   - Pixel-level metrics (sharpness, noise, specks) are measured after resampling the page to `reference_dpi`, so 150/300 DPI scans give comparable numbers.
   - Brightness thresholds for dark borders are relative to each page's own paper level.
6. **Unknown issues are flagged, not silently passed.** Each Step 2 metric has an `expected_ranges` entry, taken from the 71 sample pages. A page outside a range is flagged `unusual_<metric>`, which may mean a problem the pipeline was not designed for.
7. **No information loss: when unsure, keep it.** The 71 samples are a small part of the real collection, so every step is built to keep content it does not understand:
   - Never crop. Dark (spine/shadow) bands are **never painted**: they are flattened where recoverable and otherwise left as the original pixels.
   - Only grey (neutral) marks are ever erased as noise; coloured marks (stamps, seals, ink) never are.
   - Dark objects (photos, thumbprints, seals, black banners) keep their tones.
   - Bleed-through is lightened (60%), not erased.
   - Every page keeps its exact original scan (`{doc}_{page}_original.jpg`), and everything applied is listed in `{doc}_{page}_params.json`.
   - Planted "content trap" tests (see *Tests*) check all of this on every run.

---

## What the sample documents look like (`sample-docs/`)

| File | Pages | Landscape pages |
|---|---|---|
| FIR-1-2017.pdf | 15 | p12, p15 |
| FIR-2-2009.pdf | 17 | p13, p17 |
| FIR-22-2010.pdf | 16 | p12, p15 |
| FIR-4-2013.pdf | 23 | p19 |

- **Image-only scans:** there is no text layer. Each page is one embedded image at about **200 DPI** (≈1650–1700 × 2650–2800 px). Page sizes vary.
- **Black book-spine and scanner borders,** sometimes with part of the next page showing (FIR-22-2010 p1–4, FIR-2-2009 p2–4).
- **Coloured or aged paper:** pink/orange (FIR-2-2009 p9–10), lavender (FIR-4-2013 p9–12), yellowed sheets.
- **Faint blue ballpoint ink** and **bleed-through** of writing from the back of the sheet (FIR-2-2009 p8 is severe).
- **A grey, speckled photocopy** of a small slip placed skewed on a large white page (FIR-1-2017 p12).
- **Marks to keep:** purple stamps, thumbprints, diagonal strike-through lines (these are content).
- **Mixed page types:** printed IIF forms filled in by hand, fully handwritten statements, hand-drawn site sketches, near-blank pages, and medical/X-ray reports.

---

## Hardware & device profiles (`device.py`)

The same notebook must run unchanged on two machines. `device.py` detects which one it's on and returns a settings profile. The notebook reads that profile in Step 0.

### Target machines

| | MacBook Air M5 (development) | NVIDIA GB10 AI server (batch runs) |
|---|---|---|
| OS / CPU | macOS, Apple Silicon arm64 | Linux (DGX OS / Ubuntu), Grace ARM64, ~20 cores |
| Memory | ~24 GB unified (shared by CPU and GPU) | ~128 GB unified (shared by CPU and GPU) |
| GPU | 10-core Apple GPU (Metal) | Blackwell GPU (CUDA) |
| OpenCV GPU path | Standard `opencv-python` has **no CUDA and no Metal**. The only GPU path is OpenCV's OpenCL "T-API" (`cv2.UMat`). OpenCL is deprecated on macOS, so it must pass a self-test or we fall back to CPU. | `cv2.cuda` works **only** with OpenCV built with CUDA for this GPU (aarch64, its compute capability from `nvidia-smi --query-gpu=compute_cap`). The pip wheel has no CUDA. Alternatively use NVIDIA's OpenCV container. |
| Optional array GPU | — (MLX not used for now) | CuPy, for NumPy-heavy steps (illumination division, Sauvola, metrics) |
| Main speed lever | Process pages in parallel on the CPU cores | GPU per page + many parallel workers |

> CuPy and CUDA-built OpenCV are image-processing libraries, not AI models or OCR, so they're allowed under rule 2.

### What `device.py` detects
- **OS and architecture:** `platform.system()`, `platform.machine()`.
- **CPU:**
  - Total cores.
  - On Mac, performance-core count via `sysctl hw.perflevel0.physicalcpu`.
  - On Linux, `os.sched_getaffinity`.
- **Memory:** `sysctl hw.memsize` on Mac, `/proc/meminfo` on Linux. Also measure the memory that is free right now.
- **NVIDIA GPU:**
  - `nvidia-smi` (name, memory, compute capability).
  - `cv2.cuda.getCudaEnabledDeviceCount()`, plus a check for each CUDA function we need (denoise, filters, warpAffine, CLAHE). This moves the checks currently in notebook cell 0 into `device.py`.
- **OpenCL (Mac):** `cv2.ocl.haveOpenCL()` plus a small self-test (filter one image on GPU and CPU and compare the results).
- **CuPy:** importable and a test array runs on the GPU.

### Profiles it returns
`get_device_profile()` returns a plain dict (printed in Step 0 and saved in `metrics.json`):

| Setting | `mac_apple_silicon` | `nvidia_cuda` (GB10) | `cpu_generic` (fallback) |
|---|---|---|---|
| `backend` per operation (denoise, blur/morph, warp, CLAHE) | `opencl` if self-test passes, else `cpu` | `cuda` where available, else `cpu` | `cpu` |
| `array_backend` | `numpy` | `cupy` if available, else `numpy` | `numpy` |
| `num_workers` (parallel pages) | performance cores − 1 (≈3–4), limited by memory | ≈12–16, limited by memory | cores − 1 |
| `mem_per_worker_gb` (estimate, ≈1.5 GB per page at native resolution) | 1.5 | 1.5 | 1.5 |
| `max_render_scale` (upscale for small handwriting) | 1.0 (native only) | 1.5 | 1.0 |
| `save_intermediates` (images for every step) | off (only final + flagged pages) | on | off |
| `preview_max_px` (notebook display size) | 1400 | 1400 | 1200 |

- **Workers:** `min(cores_rule, floor(available_memory × 0.6 / mem_per_worker_gb))`, so the Mac never starts swapping. The GPU on unified memory shares that same RAM pool.
- **Overrides:**
  - Environment variable `FIR_DEVICE=cpu|mac|cuda` to force a profile.
  - `FIR_WORKERS=n` to force the worker count.
  - Any key can also be overridden in the notebook `CONFIG`.

### Same result on every device
GPU paths can differ slightly from CPU (rounding, border handling). A pipeline is only useful if the review flags mean the same thing on both machines, so:
- Each step function has one CPU reference version. The GPU version is just a faster version of the same thing.
- **`device.py` self-check:** run every step on one sample page with both CPU and the chosen GPU backend and compare.
  - If the average pixel difference is above a tolerance, that operation falls back to CPU, and the profile records the fallback.
- Fixed random seeds. Thresholds depend only on the metrics, never on the device.

### Setup per machine
- **Mac:** `pip install -r requirements.txt` (arm64 wheels, CPU + OpenCL).
- **GB10:** use the NVIDIA OpenCV/CUDA container or build OpenCV with CUDA for aarch64 against the installed CUDA, then install the rest of `requirements.txt` without `opencv-python` (so it doesn't override the CUDA build). Install `cupy-cuda13x` (or the version matching the installed CUDA) as an optional extra.
- Exact GB10 setup commands are written up when we first run on the server.

---

## Preprocessing plan (step by step)

Each step is one notebook cell. For every step we write down:

- **Why** it's needed;
- **Detect:** the measurement that decides whether it runs;
- **Method:** how it works;
- **Review flag:** when to raise the page for human review.

Steps run in this order. Light cleanup (noise removal, sharpening) comes **after** geometry fixes (orientation, borders, skew), which is the reverse of the old pipeline.

### Step 0 — Setup & configuration
- Imports. `from device import get_device_profile`, then print the detected profile (machine, backends, workers, memory budget).
- One `CONFIG` dict for all parameters and review thresholds, so tuning happens in one place. Device settings come from the profile, and `CONFIG` can override them.
- Every step function takes the profile and uses the CPU or GPU version of each operation. Pages run in parallel with `num_workers`. The notebook displays results only after all pages finish.
- Choose the input: one PDF or all of `sample-docs/`. Fix the broken sample path (`02-2009.pdf` → `FIR-2-2009.pdf`).
- Each page gets one `page_record` dict: doc, page no, images per stage, metrics per step, actions taken, review flags.

### Step 1 — Render pages at native resolution ✅ implemented
- **Why:** the scans are about 200 DPI. Rendering at 300 DPI only enlarges them and adds blur. Even rendering at "1×" resamples the image and shifts it by 1 px, because the scan sits at a 1 pt offset on the page.
- **Method:**
  - Every sample page is exactly **one embedded JPEG covering the whole page**. That image is decoded directly with pdfium (`get_bitmap(render=False)`), which gives the exact scanned pixels in BGR order with no resampling. It takes about 10–15 ms per page.
  - The pixels are **copied** out of pdfium memory (`to_numpy()` is only a view into it).
  - Pages that are not a single full-page scan (several objects, rotated or flipped placement, a PDF `/Rotate` setting) are rendered at `fallback_dpi` and flagged.
  - Optional enlargement (`CONFIG["render"]["upscale"]`, capped by the device profile). Off by default.
  - **No filtering here.** The old denoise and unsharp mask at render time are removed.
- **Output:** `images["original"]` (kept untouched for comparison). All 71 pages use about 0.9 GB of memory.
- **Metrics:**
  - extraction method, codec, pixel size, orientation;
  - `est_dpi`: the PDF says 72 DPI, which is meaningless, so this is the short side ÷ 8.5 in;
  - `color_ink_pct`: % of strongly coloured pixels;
  - `paper_tint`: median colour strength of the paper pixels;
  - `greyscale_scan`: true when the scan has no colour at all.
- **Review flags:** `low_resolution` (short side < 1240 px ≈ 150 DPI), `not_single_scan`, `page_rotation_set`, `partial_image_coverage`.
- **Result on the samples:** all 71 pages extracted directly, 0 flags. Estimated DPI is 173–200. 13 pages are pure greyscale scans; the pink pages (FIR-2-2009 p9–10) have paper tint 54–59.

### Step 2 — Page quality assessment ✅ implemented (measure only, nothing changes)
Calculates the metrics that turn the later steps on or off and stores them in `record["needs"]`. Pages run in parallel (`run_parallel`, threads; OpenCV releases the GIL): 71 pages in about 7 s on the M5.

| Metric | How | Used by |
|---|---|---|
| `borders` | Per side, on a 1/4-size copy. **Dark strips:** ≥ 3 lines with ≥ 35% dark pixels anywhere in the outer 20%; the search may skip over neighbour-page text outside a spine. **Pale strips:** lines in the outer 8% that are evenly off-paper in colour (scanner bed); these must be reached from the edge without crossing text. | Step 4 |
| `content_box` | Page minus the border bands. All metrics below are measured inside it. | all |
| `ink_pct` | Stroke pixels: black-hat (15 px) > 30 | blank detection |
| `sharpness` | High-frequency energy on stroke edges ÷ stroke contrast. Laplacian variance was rejected because it depends on how much text the page has. | Step 8 |
| `noise_sigma` | Immerkær estimate on flat pixels. Plain background deviation doesn't work because the scanner clips the paper to 255. | Step 7 |
| `texture_pct` | Faint dots not near real strokes (toner grain, paper texture) | Step 7 |
| `isolated_specks_per_mp` | Blobs ≤ 6 px with no other ink within 8 px, so dotted fill-in lines don't count | Step 7 |
| `faint_stroke_pct` | Faint, curvy, stroke-shaped marks not near real ink, after removing ruled/dotted lines. This is the bleed-through signal. | Step 9 |
| `illumination_spread` | 95th − 5th percentile of the paper level | Step 6 |
| `paper_lab`, `paper_tint` | Median Lab colour of paper pixels; chroma from neutral | Step 6 |
| `ink_level`, `ink_contrast` | Median grey level of strokes; paper − ink | Step 10 |
| `ink_colors` | Stroke share that is blue / purple / red / other / black, by chroma (not HSV saturation, which is unstable on dark pixels) | Step 6 |

- **Checking the metrics:** sharpness of real pages is 5–18, while synthetic blur scores 6–9 at σ 0.8 and 4–5 at σ 1.5. The bleed-through metric ranks FIR-2-2009 p8 (2.29) highest, and form pages with dotted lines score ≈ 0.1.
- **Review flags:** `blank`, `near_blank`, `blurry`, `noisy`, `many_specks`, `bleed_through_suspected`, `large_border` (> 8% of a side), `uneven_illumination`.
- **Result on the samples:** 12 of 71 pages flagged:
  - 6 spine/scanner borders, including the X-ray slip page and a short sheet on a larger scan;
  - 3 suspected bleed-through;
  - 2 noisy: the grey photocopy, and the spine-shadow page FIR-22-2010 p1;
  - 1 near blank.

  Pages needing each later step: border 37, tint 11, mild denoise 16, strong denoise 2, specks 8, sharpen 7, bleed 3, contrast 11, illumination 1.

### Step 3 — Orientation by content (0° / 90° / 180° / 270°) ✅ implemented
- **Why:** some scans may be sideways or upside down. Pages that are landscape *by content* must stay landscape.
- **Text mask:** strokes (black-hat) inside the Step 2 content box, at 150 DPI, with ruled and dotted lines removed (same settings as Step 2).
- **Axis (OpenCV only, no OCR):**
  - Close the mask along x and, separately, along y. Horizontal text becomes long flat blobs only when smeared along x.
  - `axis_score` = share of ink in long horizontal blobs − share in long vertical blobs.
  - On the samples it is right for all 142 cases (71 upright pages ≥ +0.02, their 90° copies ≤ −0.02).
  - *Rejected:* comparing row and column projection profiles (93%, failed on dense handwriting).
- **Up/down:**
  - Find horizontal bars (Devanagari heading bar शिरोरेखा, T-bars) and compare the ink hanging below them with the ink above them.
  - `updown_score` = vote(page) − vote(page turned 180°).
  - *Rejected:* finding the ink peak inside each text line (handwritten lines merge), and voting over all 4 rotations at once (vertical stems look like bars when the page is turned 90°).
- **Decisions (conservative):**
  - Clearly vertical text → turn 90° or 270° using the up/down vote.
  - Up/down score ≤ −0.5 → turn 180°.
  - Up/down score ≤ −0.15 → only flag `possibly_upside_down`.
  - A text-direction score between −0.10 and 0 → only flag `orientation_uncertain`.
  - **Every turned page is flagged `rotated`** so a person confirms it.
- **Output:** `images["oriented"]`. The content box and borders are carried into the new orientation.
- **Result on the samples:** 0 pages turned (all are upright), and all 7 landscape pages stay landscape. 3 pages are flagged `possibly_upside_down`: two English medical pages and the nearly blank page.
- **Self-test (real pages turned on purpose by 90/180/270°; `RUN_ORIENTATION_SELF_TEST = True`):**

  | Applied | Fixed | Turned wrong way (still flagged `rotated`) | Not turned, flagged | Not turned, **not flagged** |
  |---|---|---|---|---|
  | 90° | 58 | 11 | 2 | 0 |
  | 270° | 58 | 11 | 2 | 0 |
  | 180° | 14 | 0 | 34 | **23** |

- **Known limitation:** upside-down detection without OCR is weak on English and table pages, which have no heading bars. Sideways pages are always made horizontal, but about 1 in 6 ends up upside down; these carry the `rotated` flag. Improving the 180° cue (e.g. an ascender/descender cue for English) is open work.

### Step 4 — Border & scanner-artifact cleanup ✅ implemented
- **Why:** black spine and scanner shadows, slivers of the neighbouring page, and pale scanner-bed strips throw off the thresholds, deskew and background estimate in later steps.
- **Input:** `images["oriented"]` and the Step 2 border bands, carried through Step 3. Only pages with `needs["border_cleanup"]` are changed. Output: `images["borders_clean"]` (same size, never cropped).
- **Pale bands** (scanner bed, off-page area):
  - Fill with the page's paper colour (Step 2 `paper_lab`), with a 0.5 mm feathered edge.
  - Pen strokes that cross from the page into the band are put back. A stroke qualifies only if it:
    - reaches at most 4 mm into the band;
    - is ≤ 60 mm² in area;
    - is ≤ 0.8 mm wide at its widest point;
    - lies on paper-bright background.
- **Dark bands** (spine shadow): **flatten, never paint over.**
  - Page text is often printed *on* the shadow: row numbers, the start of field labels, "9654" on FIR-22-2010 p1.
  - The shadow level is estimated (0.4 mm blur → 3.8 mm closing) and divided out of the **lightness channel only**, so JPEG colour noise is not amplified. Leftover grain close to paper brightness is set to paper colour.
  - Only background darker than paper ÷ 6, which holds no readable text, is filled.
  - The area *beyond* the spine is flattened too, never erased. On these samples it is sometimes a neighbour-page sliver and sometimes this page's own lines continuing across the spine (FIR-22-2010 p2). Pixels alone cannot tell the two apart, so a page with ink there is flagged `content_beyond_spine` for a person to decide.
- **Verify:** Step 2's border detector runs again on the result. A remaining **dark** strip is flagged `border_remaining`; remaining pale bands are recorded only.
- **Approaches tried and rejected (all damaged content):**
  - Filling the whole dark band: erased text printed on the shadow.
  - Restoring every ink blob crossing the edge: brought the spine back, because spine texture joins up with text.
  - Splitting at the darkest line of the spine: cut "9654".
  - Erasing everything beyond the spine: cut the ends of lines written across it.
  - Stronger grain clean-up (70% of paper level, 0.4 mm blur): erased faint text on the shadow.
- **Review flags:** `text_at_border` (> 20 mm² of strokes kept inside a band), `content_beyond_spine`, `large_fill` (> 20% filled), `border_remaining`.
- **Result on the samples:** 37 pages cleaned in about 2 s. 8 pages flagged:
  - 6 `content_beyond_spine`: the spine pages of FIR-2-2009 and FIR-22-2010;
  - 5 `border_remaining`: the black lid strips at the top/bottom of FIR-22-2010 and a thin right-edge line on FIR-2-2009 p1. Grain left in the flattened spines goes to Step 7.

### Step 5 — Deskew (small angle, ±10°) ✅ implemented
- **Input:** `images["borders_clean"]` and the content box. Measured on a 150 DPI copy of the content box only. Output: `images["deskewed"]`.
- **Two independent estimates:**
  1. **Lines (Hough):** straight segments ≥ 30 mm (form rules, dotted lines with ≤ 1.5 mm gaps, and also the long Devanagari heading bars in handwriting), within ±10°, combined as a length-weighted median.
  2. **Projection:** the stroke mask with ruled lines removed is rotated through ±10° (coarse 0.5°, then fine 0.05°), and the angle with the sharpest row profile wins. Rotation uses **nearest-neighbour** on purpose: linear interpolation blurred every angle except 0° and biased the search to "no skew".
- **Decision, with thresholds measured by the self-test** (real pages tilted by ±0.6/1.5/4°):
  - Strong lines (≥ 10 segments and ≥ 1 m total) → line angle. Self-test: median error 0.09°, 95% within 0.48°.
  - Otherwise strong projection (peak ≥ 1.1× sharper than at 0°), unless even weak lines differ by > 1° → projection angle. Self-test: median error 0.10°.
  - Otherwise both weak but within 0.5° → line angle. Otherwise **not rotated**.
- **Rotation:** about the centre with an expanded canvas, filled with paper colour, cubic interpolation, on the device's warp backend (OpenCL/CUDA, CPU fallback). Portrait/landscape never changes.
- **Slanted handwriting:** some writers' lines rise 5–6° on a straight sheet (FIR-1-2017 p9). The step levels the text, because that is what matters for extraction, and flags it `large_skew`.
- **Review flags:**
  - `deskew_disagree`: both estimates strong but different (e.g. a printed table vs the handwriting on it);
  - `large_skew` (> 5°);
  - `deskew_uncertain`: no trustworthy estimate, but one suggests ≥ 0.5°; not rotated.
- **Result on the samples:** 43 pages rotated (median |angle| 0.69°, max 6.4°), all 7 landscape pages stay landscape. 8 pages flagged:
  - 6 `deskew_uncertain`: dense or slanted handwriting with weak evidence;
  - 1 `deskew_disagree`: FIR-2-2009 p13, table vs handwriting;
  - 1 `large_skew`: FIR-1-2017 p9.

  Visual check: tables and form lines are level, and handwriting is level on the handwritten pages.
- **Known limit:** sparse or slanted-handwriting pages have no single "right" angle without OCR. They are left alone and flagged rather than guessed.

### Step 6 — Illumination & paper-colour normalization ✅ implemented
- **Input:** `images["deskewed"]`. Output: `images["illum_norm"]`.
- **Local white balance:**
  - Estimate the paper colour per channel on a 50 DPI copy, using a 12 mm closing (wider than strokes, gaps in dense handwriting and black banner bars), then 12 mm median + Gaussian smoothing.
  - Divide it out: `out = image × 255 / paper`. Shadows, yellowing and pink/lavender paper all become white, and ink keeps its colour relative to the paper.
- **Safety:**
  - Gain limited to 0.95–1.5×, so thumbprints and big stamps wider than the window are not bleached. If more than 1% of the page hits the cap, it is flagged `gain_capped`.
  - Soft white point on **luminance**, 235–255. The first version used the brightest channel, which washed out blue ink (−10 to −25% strokes on blue-ink pages).
- **Applied to every non-blank page** (`apply_to_all_pages`), so all pages share one white. `needs` from Step 2 is recorded only.
- **Verified per page (before → after):**
  - paper lightness, tint and spread;
  - stroke-pixel count (`ink_loss` if it drops > 10%);
  - blue/purple/red share of strokes (`colour_loss` if a share *drops* > 10 points; a rise is expected, because blue ink on yellow/pink paper regains its true colour, e.g. 43% → 66% on FIR-2-2009 p8);
  - `tint_remaining`, `illumination_remaining`.
- **Result on the samples (71 pages, about 14 s):**
  - paper lightness median 99.6 → 100;
  - tint worst 19.6 → 0 (pink pages);
  - spread worst 73 → 17;
  - black banner bars and thumbprints intact.

  1 page flagged: the grey photocopy slip FIR-1-2017 p12. Its grain was counted as "ink" before and is lighter now; the rest of the grain goes to Step 7.

### Step 7 — Noise removal (adaptive) ✅ implemented
- **Input:** `images["illum_norm"]`. Output: `images["denoised"]`.
- **Detect on the current image:** Step 2's `assess_page` runs again, because Step 4's flattening and Step 6's white balance change the noise picture. Each fix runs only where its measurement says so:
  - **NLM on lightness only** (Lab L): mild (h = 6) at noise σ ≥ 1.0 or texture ≥ 2%, strong (h = 12) at σ ≥ 2.0. Patch 0.9 mm, search 2.7 mm, on the device's denoise backend.
  - **Colour speckle:** 0.6 mm median on Lab a/b, on colour pages that get NLM.
  - **Texture dots → paper:** at texture ≥ 0.5%. These are the faint *dot-like* marks away from strokes that Step 2 counts as texture. Ruled/dotted lines and faint *stroke-shaped* marks are kept (Step 9's job).
  - **Isolated specks → paper:** at ≥ 100/MP. Blobs ≤ 0.1 mm² with no ink within ~2 mm; dotted lines, full stops and matras have neighbours.
- **Verify:** `assess_page` runs again after the fixes. Ink loss is measured on **real strokes only** (connected ink ≥ 0.5 mm²), because removing grain is the point.
- **Review flags:** `ink_loss` (> 8%), `noise_remaining` (σ ≥ 2), `specks_remaining`.
- **Result on the samples:** 24 pages cleaned in about 13 s:
  - texture down to ≤ 0.5% everywhere (e.g. FIR-2-2009 p12: 1.28% → 0.02%);
  - specks e.g. 206 → 18/MP;
  - real strokes within ±5% on all pages but two.

  4 flags on 3 pages:
  - FIR-1-2017 p12: its grain forms large clusters that count as strokes; the crops confirm the text is intact;
  - FIR-4-2013 p20: −11.6% from smoother edges on dense faint blue writing;
  - FIR-22-2010 p1/p3: `noise_remaining`.
- **Known limit, tried and reverted:** the coarse grain left in spine shadows flattened by Step 4 is not removed. NLM barely changes it. Removing small clusters inside Step 4's flattened zone also failed: the grain joins into larger clusters and survives, and the dots of dotted fill-in lines beside the spine were erased.

### Step 8 — Deblur / sharpen (adaptive) ✅ implemented
- **Input:** `images["denoised"]`. Output: `images["sharpened"]`.
- **Detect on the current image:** sharpness is measured again with `assess_page`. Pages ≥ 7.0 (`sharpen_below`) are left alone.
- **Method (unsharp mask on Lab L only, ink colour untouched):**
  - `detail = L − Gaussian(L, 0.2 mm)`.
  - Soft threshold: detail < 4 grey levels is not boosted (paper and residual noise).
  - Strength from the deficit: `amount = 3 × (9 − sharpness) / 9`, clipped to 0.3–1.5.
  - **Halo clamp:** each pixel stays within its 3×3 min/max ± 8 grey levels.
- **Verify:** sharpness, noise and real-stroke pixels before → after.
- **Review flags:** `still_blurry` (< 5.0 after), `noise_amplified` (σ +0.5), `ink_change` (real strokes ±10%).
- **Result on the samples:**
  - 6 pages sharpened, the soft blue-ink handwriting of FIR-1-2017 p9–11 and FIR-2-2009 p7/p8/p10: sharpness 5.2–6.9 → 6.8–8.1, noise +0.1–0.35, real strokes −6.5…+5.5%;
  - 0 flags; zoom crops show crisper strokes and no halos;
  - the other 65 pages were already sharp and are untouched.
- **Self-test** (`RUN_SHARPEN_SELF_TEST`: sharp pages blurred on purpose): unsharp masking recovers about 20–30% of the sharpness lost to a strong 0.2 mm blur. Lighter 0.1 mm blur stays above the threshold and is left alone. True deblurring (deconvolution) is not attempted: it amplifies noise and is risky on handwriting.

### Step 9 — Bleed-through suppression ✅ implemented
- **Input:** `images["sharpened"]`. Output: `images["bleed_clean"]`.
- **What bleed-through looks like here:** the mirrored back-side writing is **the same ink colour as the front writing, only lighter**. On FIR-2-2009 p8 the front blue has a mean lightness of about 174 and the show-through about 214. It is *not* "faint" by Step 2's stroke measure (it has clear edges), and after Steps 6–7 that measure fell below its threshold on every page. So the faint-stroke trigger is reported for comparison only.
- **Detect (every page):**
  - Ink components ≥ 0.3 mm² are grouped by colour (blue/purple/red).
  - Each group's lightness is split with Otsu. It counts as bleed-through only if the split is clean: η ≥ 0.8, gap ≥ 30 L, light share 25–85%, ≥ 100 components.
  - Black/grey ink is not tested: print and grey dotted lines always split.
  - On the samples: the bleed pages have η 0.88–0.92 and a gap of 39–58. Normal blue pages have η 0.47–0.69, or a small light share.
- **Fix (per pixel):**
  - Pixels of that colour that are lighter than the split are blended 85% toward paper: lightened, not erased.
  - Pixels within 0.25 mm of dark ink of that colour (front strokes) or of black/grey ink (print) are never touched.
- **Rejected approaches:**
  - Classifying whole connected components: bleed touches front strokes and print, so they merged. A dark front stroke on p8 was lightened and **printed text on p9 was erased**.
  - A 15% minimum light share: it caught faint real handwriting on FIR-4-2013 p5/p18.
- **Review flags:** every changed page is flagged `bleed_suppressed`. Also `ink_change` (front strokes ±5%), and `bleed_not_confirmed` when Step 2 suspected bleed-through but no clean split exists (FIR-1-2017 p10, which is ink spreading into the paper).
- **Result on the samples:** 3 pages (FIR-2-2009 p5, p8, p9). 1.2–5.9% of each page lightened, front strokes changed ≤ 0.1%, and the crops show print and dark handwriting intact.
- **Limits:**
  - Faint ghost outlines of the show-through remain.
  - Show-through overlapping the front writing stays.
  - Black-ink bleed-through is not handled.

### Step 10 — Faint ink strengthening ✅ (revised: detect per mark)
- **Why the first version missed it:** it looked at the **median** darkness of all ink. On form pages the dark handwriting kept the median "dark enough" (e.g. 164 on FIR-2-2009 p1/p4), so light-grey printed labels ("Act", "Sections", "Action taken…") were never strengthened.
- **Detect (per mark):**
  - Every connected stroke's **core** (darkest pixel) is measured: dark handwriting ≈ 30–90, light printed toner or faded ink ≈ 130–215.
  - Faint marks are those with a core above 110 and an area of at least 0.1 mm².
  - The step runs only if faint marks make up ≥ 10% of the ink area. On the samples that's 38 pages, including FIR-2-2009 p1 (27%) and p4 (31%); clean pages sit at 2–5%.
- **Fix:**
  - A monotone, smoothed tone curve through (0, 0), (faint core → 80), (240 → 240), (255 → 255), on lightness only.
  - It is applied **only to the faint marks** (+1 px soft edge). Paper, dark handwriting, photos and stamps keep their exact pixels.
  - Monotone means no stroke can disappear; colour is unchanged.
  - On show-through pages (Step 9 applied) only **grey** faint marks are strengthened, so lightened show-through is not darkened again.
- **Rejected on the way:**
  - A whole-page curve: the self-test caught a dark photo flattened (detail 32 → 22).
  - Measuring with the weak stroke threshold: it anchored the curve near paper and barely darkened anything.
  - Faint core → 110: still too light.
- **Result:** FIR-2-2009 p4 printed text mean grey 194 → 155, darkest tenth 140 → 65. All 7 self-test traps pass. One flag: FIR-1-2017 p12 `noise_amplified` (the grainy photocopy slip).

### Step 11 — Final outputs ✅ implemented
- **Files:** see *Output naming* under *Code review*. `{doc}_{page}_original.jpg`, `_cleaned.png`, `_cleaned_bw.png`, `_params.json` per page; `{doc}_metrics.json` per document; `summary.json`. Stage images (`{doc}_{page}_stageNN_<stage>.jpg`) only when `save_stages` / the GB10 profile.
- **Black & white:**
  - Sauvola `T = mean × (1 + k × (std/R − 1))`, 6 mm window, k = 0.15, R = 128, written in NumPy/OpenCV with box filters. Isolated blobs < 0.02 mm² are removed.
  - Computed on the **darkest channel** (min of B, G, R): faint blue ink is much darker in red than in luminance.
  - Measured on all pages: 98.8% of stroke cores kept (median), versus 92.7% for luminance with k = 0.2, which broke faint blue handwriting on FIR-1-2017 p13. Noise stays ≤ 0.15%.
- **Checks:**
  - `bw_ink_recall`: share of stroke **cores** (black-hat > 45 inside real strokes) that are black. Flag `bw_ink_loss` < 85%.
  - `bw_noise`: black pixels > 0.5 mm from any stroke. Flag at > 1%.
  - The first version counted soft stroke edges too, which put every page near 80% and flagged all of them.
- **Result on the samples:** 142 PNGs (193 MB) plus JSON in about 8 s. Ink recall median 0.988; B&W noise ≤ 0.15%. 1 flag: FIR-1-2017 p12, the grey photocopy, where grain turns into speckle and the recall is 0.69.
- **Pages needing review, all steps together:** FIR-1-2017 5/15, FIR-2-2009 11/17, FIR-22-2010 4/16, FIR-4-2013 6/23 (from `summary.json`).

### Step 12 — Quality check & review filter
- Run the Step 2 metrics again on the final images. Record before → after for:
  - blur;
  - noise;
  - background whiteness;
  - text contrast;
  - remaining skew;
  - ink preserved.
- Show a **summary table** for all pages: metrics, actions taken, flags.
- **Review filter cell:**
  - Pick a flag (e.g. `rotated`, `deskew_disagree`, `bleed_through`, `ink_loss`) or show all flagged pages.
  - Each one is shown as Original | Cleaned | difference overlay.
- Pages with no flags can be accepted in bulk. Flagged pages are checked by eye and their thresholds adjusted in `CONFIG`.

---

## Code review for information safety (large collections)

Reviewed after Step 11 for operations that could remove content on documents unlike the samples:

| Step | Risk found | Change |
|---|---|---|
| 4 Borders | Dark-band parts too dark to flatten were painted with paper: could hide white-on-black content | Left as original (`fill_unrecoverable_dark: False`); white-on-black marks are flagged `inverted_content_in_border` |
| 6 Lighting | Dark objects > 12 mm (photo, thumbprint, seal, banner) were taken for dark paper and brightened up to 1.5× | Paper estimate is inpainted from the surrounding paper there; flag `dark_object_kept` |
| 6 White point | Easing to white started at luminance 235: faded pencil and light stamps | Starts at 245 |
| 7 Texture/specks | Faint stamps and coloured dots could be erased as texture/specks | Only neutral (grey) marks are cleared; texture only if faint |
| 7 Colour median | Thin coloured strokes turned grey | Applied to paper pixels only |
| 9 Bleed-through | 85% lightening nearly erased misdetected light writing | 60% |
| All | Every stage of every page kept in memory (~100 MB/page) | `process_documents()` batch mode streams pages and frees images |

**Spine shadow (reviewed on FIR-22-2010 p1/p2):**
- Default: flattened only (`shadow_lift: None`). All text on both sides of the spine is kept.
- Optional `shadow_lift: (0.45, 0.80)`: a monotone tone curve that lightens the leftover grain without removing strokes. Also acceptable.
- (0.40, 0.70) was too strong.
- Binary text/background separation erased characters ("9" of "9654", "(i)–(iv)", "Type of") and is not used.

*Rejected on the way:* a band-level "white-on-black found → leave the whole band untouched" guard. The paper between dense dark letters looked like white glyphs, so every spine page was left uncleaned. It is now a solid-dark-background test (≥ 60% black within 4 mm) and a review flag only.

### Tests (last notebook cell, `RUN_TESTS`)
Content traps are planted on a real clean page (FIR-4-2013 p4) and run through Steps 2–10:
- white-on-black banner at the top edge;
- dark 25 × 30 mm photo;
- faint grainy purple stamp;
- faint pencil (L ≈ 215);
- 1 px blue line;
- light-blue-only writing;
- plus a second run to check the output is deterministic.

**All 7 pass:** banner contrast 214 → 203, photo mean 64 → 65 and std 32 → 33, stamp 100% still coloured, pencil 100% visible, blue line chroma 100%, light blue not treated as bleed-through, identical output on the second run. The cell `assert`s, so a regression stops the notebook.

### Output naming (Step 11)
`output/<doc>/`:
- `{doc}_{page}_original.jpg`: exact embedded scan, byte for byte (decoding it gives the pipeline's input exactly);
- `{doc}_{page}_cleaned.png`;
- `{doc}_{page}_cleaned_bw.png`;
- `{doc}_{page}_params.json`: the 11 steps **in pipeline order**, each with applied yes/no, why, the parameters actually used (e.g. deskew −0.61°, NLM h = 12, gamma 2.1), the result and its flags;
- `{doc}_metrics.json`, plus `output/summary.json`.

Pages are zero-padded to 3 digits (more for longer documents), so files sort in page order.

### Batch mode
`process_documents(pdf_paths)` runs Steps 1–11 page by page, `pages_in_flight` (default 2 × workers) at a time. Each page's files are written and its images freed; output is identical to the step-by-step run.

---

## Deferred — page classification

The page-labelling work restarts after preprocessing is approved. Notes for later:
- **Label set:** IIF-1 FIR, IIF-2 Crime Details, IIF-3 Arrest/Court Surrender, IIF-4 Seizure, IIF-5 Final Report, Medical / X-ray / Injury, handwritten statement, site sketch, blank.
- **Strongest printed clue:** the top-right corner mark `I.I.F.-x / एकीकृत जाँच फार्म-x, पृष्ठ सं.-n` (form and page number within the form).
- **Must follow the no-third-party-AI/OCR rule.** The archived cells used Surya OCR v2, which will have to be replaced.

---

## Usage (`image_processing.py`)

```python
from image_processing import DocumentPreprocessor, preprocess_pdf

pre = DocumentPreprocessor()                         # settings + device detected once; config={...}, device="mac"/"cuda"/"cpu"
finals = pre.process("doc.pdf")                      # list[np.ndarray] RGB, one per page, page order (color="bgr" for OpenCV)
pages  = pre.process("doc.pdf", return_all=True)     # list[PageResult]: .page .original .final .applied .files
pre.process("doc.pdf", save=True)                    # also save, in the current directory
pre.process("doc.pdf", save=True, output_dir="out")  # ... or in out/
for page in pre.iter_pages("big.pdf"):               # one page at a time, for very large PDFs
    ...
pre.self_test()                                      # planted-content information-preservation test
finals = preprocess_pdf("doc.pdf", save=True)        # one-line shortcut
```

Command line: `python image_processing.py doc1.pdf doc2.pdf -o out` (`--no-save`, `--device mac|cuda|cpu`).

**Saved per page** (flat in the output folder, no per-PDF folder; page zero-padded to 3 digits):

| File | What |
|---|---|
| `{pdf}_{page}_original.jpg` | Exact scan from the PDF, byte for byte (`.png` if the page had to be rendered) |
| `{pdf}_{page}_final.png` | Cleaned colour page, **lossless**. This is the image to send to the model |
| `{pdf}_{page}_applied.json` | Every step in pipeline order: applied yes/no, why, parameters used, result, flags, `needs_review` |

**Which image to send to a model:**
- Send `final` straight from memory (an uncompressed array), or the saved PNG, which is identical.
- Use colour, not black-and-white: colour keeps faint ink, stamps, and blue vs. black ink.
- Black & white is off by default (`CONFIG["output"]["bw"]["enabled"]`).

**Verified:**
- The module's final images are pixel-identical to the notebook's step-by-step output on all 71 sample pages.
- Original JPEGs are byte-identical to the PDF.
- `self_test()` passes.
- About 90 s for 71 pages on the M5.

---

## Getting started

Requires **Python 3.12+**. On macOS the default `python3` (Xcode, 3.9) is too old for NumPy 2.5 / OpenCV 5, so use Homebrew's:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook app.ipynb
```

Run the cells top to bottom. Each step prints its per-page metrics and what it did.

## Project structure

- [image_processing.py](image_processing.py): the whole preprocessing pipeline and its API (`DocumentPreprocessor`, `preprocess_pdf`, CLI).
- [app.ipynb](app.ipynb): review notebook. It runs the module's steps one by one with tables and before/after views.
- [device.py](device.py): detects Mac M5 / NVIDIA GB10 / generic CPU and returns the settings profile. Run `python device.py` to print it.
- [sample-docs/](sample-docs/): sample scanned FIR case files.
- [archive/page_classification_cells.py](archive/page_classification_cells.py): removed classification cells, kept for reference (not run).
- [requirements.txt](requirements.txt): Python dependencies.
