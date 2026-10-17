# FoodGuard AI

Food-safety detection for Indian consumers. Upload a photo and get one clear verdict: **SAFE / CAUTION / UNSAFE**, or **REJECTED / MISMATCH** when the image is not usable.

* **Raw produce**: a fine-tuned MobileNetV2 detects spoilage, and Grad-CAM shows where it looked.
* **Packaged food**: EasyOCR reads the nutrition panel, and the label is checked against FSSAI HFSS limits (red-hexagon warnings). An expiry date can be entered by hand if you want it checked too; it is optional.
* **Guardrails**: three checks run before analysis. Is it food? Does it match the chosen category? Is it the product you named?
* **Review**: every analysed item gets a 0–100 health score with a rating and star display, plus information about the food (nutrition, storage, shelf life, signs of spoilage, share of the daily maximum).

```
                    [ User uploads image(s) + category (+ optional name / barcode / expiry date) ]
                                              |
              Stage 1  Out-of-food filter (CLIP zero-shot)  ----- fail --> REJECTED "Not a food item"
                                              |
              Stage 2  Category validation (CLIP + OCR cues) ----- fail --> MISMATCH "Category mismatch"
                         raw vs packaged / solid vs liquid / unsupported produce / no nutrition panel
                                              |
              Stage 3  Product identity (barcode, Open Food Facts, brand families, nutrient cross-check)
                                              |                       ----- fail --> MISMATCH "Wrong product"
                 +----------------------------+-----------------------------+
                 v                                                          v
     Pipeline A: RAW PRODUCE                                   Pipeline B: PACKAGED FOOD
     MobileNetV2 (12 classes)                                  OpenCV pre-processing -> EasyOCR (multi-variant)
     P(fresh) >= .85 -> SAFE  (Grade A)                        Regex + RapidFuzz parser -> per 100 g / 100 ml
     P(stale) >= .85 -> UNSAFE (Grade D)                       FSSAI limits per category (+ expiry, if entered)
     otherwise       -> CAUTION (Grade B)                      expired or any limit exceeded -> UNSAFE + red hexagons
     + Grad-CAM heatmap + discolouration %                     unreadable nutrient -> CAUTION; else SAFE
                 |                                                          |
                 +------------> Review: health score 0-100, rating, food info <-----+
```

---

## 1. Setup on a new PC

You need Python 3.10 or newer (3.12 was used) and Git. The trained spoilage model (`models/spoilage_mobilenetv2.pt`) is included in the repository, so **no training is needed** to run the app.

```powershell
git clone https://github.com/OMx0777/FOOD.git
cd FOOD
python -m venv .venv
.venv\Scripts\activate

# 1. PyTorch. Pick ONE line:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU (RTX 50-series needs cu128)
pip install torch torchvision                                                     # no NVIDIA GPU (runs on CPU, slower)

# 2. Everything else
pip install -r requirements.txt

# 3. Run
python app.py        # open http://127.0.0.1:5000
```

The first run needs internet: it downloads CLIP (~600 MB) and the EasyOCR weights (~100 MB). After that the app starts offline; only product look-ups use Open Food Facts.

Not in the repository: the 1.5 GB produce dataset (`data_raw/`, `dataset_vision/`). It is only needed to retrain or re-evaluate the model; `python scripts/download_datasets.py` downloads and splits it again.

The project report is in `docs/FoodGuard_AI_Project_Report.docx`.

## 2. Data and training (already done in this folder)

```powershell
python scripts/download_datasets.py      # Kaggle download (no API key needed) + leakage-safe split
python scripts/train_vision.py           # ~3 min on an RTX 5060 Ti; also evaluates on the test split
python scripts/evaluate_vision.py        # re-run the test evaluation  -> models/vision_metrics_test.json
python scripts/make_synthetic_labels.py  # synthetic labels with known values for the OCR bench
python scripts/evaluate_ocr.py           # OCR test bench               -> models/ocr_metrics.json
python -m pytest                         # 49 unit tests (parser, FSSAI rules, identity, review)
```

## 3. Run the web app

```powershell
python app.py        # open http://127.0.0.1:5000
```

| Endpoint | Purpose |
|---|---|
| `GET /` | Web interface |
| `POST /api/analyze` | multipart: `images` (1–3 files), `category`, optional `product_name`, `barcode`, `expiry_date` (YYYY-MM-DD) |
| `GET /api/health` | device, which models are loaded |

Categories: `raw_produce`, `packaged_solid`, `packaged_liquid`, `packaged_exempt`.

### Using the web page

1. Pick what you are checking: *Fruit or vegetable*, *Packaged food*, *Packaged drink*, or *Oil, ghee or honey*.
2. Drop in a photo. For produce, one photo of a single item filling the frame. For packs, up to 3 photos with the nutrition table first.
3. Optionally open **Add details** to give the product name, the barcode, or the expiry date.
4. Press **Check this food**. A scanning animation runs over your photo while the steps tick off.
5. The result shows a stamped seal with the health score, a plain-language verdict, the FSSAI red hexagons if any limit is exceeded, information about the food, how the score was worked out, every check that ran, and the Grad-CAM or OCR image.

The sun / moon button in the top corner switches between the light theme (white, off-white and green) and the dark theme. The choice is remembered in the browser.

---

## 4. Project structure

```
FoodGuard_AI/
├── app.py                       Flask server
├── foodguard/
│   ├── config.py                paths, categories, supported produce
│   ├── models.py                lazy, thread-safe loaders (CLIP, EasyOCR, MobileNetV2)
│   ├── guardrails.py            Stage 1 + 2: CLIP zero-shot scene & produce identification
│   ├── vision.py                Pipeline A: spoilage CNN, Grad-CAM, HSV discolouration
│   ├── ocr_engine.py            Pipeline B: OpenCV pre-processing, EasyOCR, row grouping
│   ├── label_parser.py          nutrients, units, per-100 normalisation, ingredients
│   ├── fssai_engine.py          HFSS limits per category, red hexagons, exempt detection, optional expiry
│   ├── review.py                health score, rating and food information
│   ├── data/produce_info.json   nutrition and storage facts for the 6 produce types (USDA FDC)
│   ├── identity.py              Stage 3: barcode, Open Food Facts, brand families
│   └── pipeline.py              orchestrator -> one verdict
├── fssai_config/
│   ├── safety_thresholds.json   ALL limits and thresholds (no magic numbers in code)
│   └── brand_registry.json      brand -> parent company (Lays -> PepsiCo ...)
├── dataset_vision/              train / validation / test (+ split_report.json)
├── dataset_ocr/
│   ├── raw_images/              label photos (synthetic_* + your own)
│   └── ground_truth_labels.json true per-100 values for the OCR bench
├── models/                      weights, class names, metrics, confusion matrix, training log
├── scripts/                     download / train / evaluate / synthetic labels
├── templates/, static/          web UI (Young Serif + Figtree fonts, light / dark themes)
├── docs/                        FoodGuard_AI_Project_Report.docx (project report)
└── tests/                       pytest suite
```

---

## 5. Food categories and thresholds

All values live in `fssai_config/safety_thresholds.json`.

| Category | Basis | Added sugar | Saturated fat | Sodium |
|---|---|---|---|---|
| Packaged – Solid | per 100 g | > 3.0 g | > 4.2 g | > 0.625 g |
| Packaged – Liquid | per 100 ml | > 1.5 g | > 2.1 g | > 0.300 g |
| Packaged – Exempt (oil, ghee, honey, dried fruit, single-ingredient) | – | exempt | exempt | exempt |

> ⚠ **Before submission, verify these numbers against the current FSSAI notification and cite the exact clause.** The values came from the project brief. FSSAI's front-of-pack labelling (FOPNL / HFSS) rules have been published as drafts with energy-percentage criteria, so an examiner may ask where each number comes from. Changing the JSON is enough. No code changes are needed.

Rules the engine applies:

* **Salt → sodium**: sodium = salt ÷ 2.5. mg → g conversion is automatic.
* **Added sugar not declared**: total sugars are used instead (conservative), and the UI says so.
* **"< 0.5 g" values**: compared as an upper bound. A value like "< 5 g" against a 4.2 g limit is reported as *uncertain*.
* **Exempt suggestion**: if the product title says ghee, honey, mustard oil and so on, or the ingredient list has 1–2 items matching those words, the UI offers a one-click "re-run as Exempt". Snack words such as "Butter **Cookies**" block the suggestion.
* **Verdict**: any limit exceeded → UNSAFE with red hexagons. A nutrient that could not be read → CAUTION. Otherwise SAFE.
* **Expiry date (optional)**: the user may type it in. A date that has passed → UNSAFE. No date entered never lowers the verdict; dates are not read from the photo.

## 5a. Health score and review

| Food | How the 0–100 score is made |
|---|---|
| Fruit or vegetable | Freshness confidence of the spoilage CNN: P(fresh) × 100 |
| Packaged | Starts at 100. Each HFSS nutrient loses 0 points up to half its limit, 12 at the limit, up to 35 at three times the limit. Any limit exceeded caps the score at 45, so the score always agrees with an UNSAFE verdict. An expired pack (date entered) scores 0. |
| Exempt (oil, ghee, honey) | Not scored |

Ratings: 85+ Excellent, 70+ Good, 50+ Fair, 30+ Poor, below 30 Avoid. Stars are score ÷ 20 in half steps.

Food information: for produce, nutrition per 100 g, what it is known for, how to store it, how long it keeps and signs it has gone off (`foodguard/data/produce_info.json`, USDA FoodData Central values). For packs, the nutrients read from the label, the ingredients, and each nutrient's share of the WHO adult daily maximum (2,000 kcal, 50 g free sugar, 22 g saturated fat, 2 g sodium).

---

## 6. How each guardrail works

**Stage 1: out-of-food filter.** CLIP (ViT-B/32) scores the photo against 43 text prompts grouped into *raw produce / packaged solid / packaged liquid / exempt / prepared food / non-food*. If the summed food probability is below 0.40, the photo is rejected. Cooked dishes are rejected with their own message.

**Stage 2: category validation.**
* Raw vs packaged: if the CLIP group probabilities differ by more than 0.30, the result is MISMATCH with a suggested category and a one-click re-run.
* Unsupported produce: the CNN only knows 6 types and would confidently "grade" a mango as an apple. If CLIP puts less than 0.10 total probability on the six supported types, the photo is rejected as unsupported.
* Packaged: at least 6 text regions and at least 2 recognised nutrition rows are required. Otherwise: "upload the back-of-pack nutrition panel".
* Solid vs liquid: the printed basis decides ("per 100 ml" means liquid, "per 100 g" means solid).

**Stage 3: identity.**
* Barcode: decoded from the photo with the OpenCV detector or typed by the user. Two different codes → MISMATCH.
* Reference record from Open Food Facts, by barcode or by name search. Among the best name matches, a record that has nutrition data is preferred.
* Brand families: brands printed on the pack are matched against the named product. A brand from a *different* company → MISMATCH. A parent company is fine ("PepsiCo" on a Lays pack).
* Nutrient cross-check: label vs reference. If ≥3 nutrients disagree (and that is ≥60% of those compared) → MISMATCH. If 2 disagree → DIVERGENT warning.
* Raw produce: the typed name (English or Hindi, e.g. *tamatar*, *kela*) must match what CLIP or the CNN see.

---

## 7. Measured results (all reproducible with the scripts above)

### Spoilage CNN: held-out test split (2,208 images)

The Kaggle dataset is heavily augmented: one source photo appears up to 164 times as rotated, flipped or noisy copies. `download_datasets.py` recovers each file's **source photo** and splits *by photo*, so no test image has a sibling in the training set. Without this, the test accuracy would be inflated.

| Metric | Value |
|---|---|
| 12-class accuracy | **97.2 %** (macro F1 0.967) |
| Fresh-vs-stale accuracy | **97.1 %** |
| Decision rule: SAFE / CAUTION / UNSAFE | 929 / 222 / 1057 |
| Accuracy on decided cases (SAFE or UNSAFE) | **99.3 %** |
| **Stale produce marked SAFE** | **0** |
| Fresh produce marked UNSAFE | 13 |

Weakest classes: fresh tomato (recall 0.78, confused with stale tomato) and capsicum. See `models/confusion_matrix_test.png`. Note that the vegetable classes come from only 8–32 distinct source photos each.

### Guardrails (CLIP)

Calibrated on 740 in-distribution test images and 54 out-of-distribution Wikimedia photos.

| Check | In-distribution false rejections | Out-of-distribution caught |
|---|---|---|
| Non-food (cats, shoes, phones, cars, people, shampoo ...) | 0.4 % | 23 / 24 at Stage 1; the 24th (a shampoo bottle) is rejected at Stage 2 for having no nutrition panel |
| Raw vs packaged mismatch | 0.9 % | – |
| Unsupported produce (mango, onion, carrot, pomegranate ...) | 1.2 % | 17 / 22 (77 %) |

### OCR + FSSAI parser

* **Synthetic bench** (6 labels with photo distortions and glare): 100 % per-field and verdict accuracy. This shows the logic is correct on clean print, not real-world accuracy.
* **20 real Indian label photos from Open Food Facts** (Parle-G, Lays, Maggi, Kellogg's, Good Day, Snickers ...): 6 were correctly rejected as unreadable (low resolution or no table). Of the values the system did read, **76 %** match the reference within 15 %. The reference itself is crowd-sourced and sometimes wrong or from a different pack version (e.g. Tata Salt is listed with 0 g sodium).
* OCR repairs that made a measurable difference: reading the unfiltered colour image first (CLAHE was *losing* decimal points), voting across 2–4 image variants, restoring lost decimals from the per-serve column, treating "(g)" read as "9" as a unit, and a physical-plausibility check (e.g. 3,000 kcal/100 g is impossible) before scaling per-serve tables.

---

## 8. Custom test bench (do this before the viva)

1. Photograph 30–50 real packs (Lays, Maggi, Amul butter, Britannia, Kurkure, juices ...) on a phone, back panel, filling the frame.
2. Put them in `dataset_ocr/raw_images/` and add one entry each to `ground_truth_labels.json`, using the values *printed* on the pack.
3. Run `python scripts/evaluate_ocr.py --real-only` and report the per-field accuracy and verdict agreement.
4. Do the same for produce: 50–100 phone photos of local tomatoes, bananas and so on. Upload them through the UI, or add them to `dataset_vision/test/<class>/` and re-run `evaluate_vision.py`.

---

## 9. Likely viva questions

* **"Why not just use accuracy on a random split?"** The dataset is augmented copies of a few hundred photos. A random split leaks near-duplicates into the test set, so we split by source photo (`split_report.json`).
* **"How do you avoid false red alerts on butter or juice?"** Category-specific limits (solid per 100 g vs liquid per 100 ml) and an exempt category for single-ingredient foods, with automatic suggestion.
* **"What if someone uploads a shoe, or a mango?"** CLIP zero-shot OOD filter (Stage 1) and a supported-produce check (Stage 2), with measured rejection rates.
* **"What if the photo is of a different product?"** Stage 3: barcode, Open Food Facts reference, brand families and nutrient cross-check.
* **"Why 0.85?"** It's in the config. On the test split it gives zero stale→SAFE errors while sending only 10 % of photos to CAUTION.
* **"Why is the CNN trustworthy?"** The Grad-CAM heatmap shows the regions that drove each decision.

## 10. Known limitations

* OCR on small, curved, glossy or low-resolution print is the main source of error. The system answers CAUTION, or REJECTED with "retake the photo", rather than guessing.
* Name-based identity depends on crowd-sourced Open Food Facts data. Barcode-based identity is much more reliable.
* The spoilage model covers 6 produce types. Other produce is rejected, not graded.
* Expiry dates are not read from the photo (inkjet and dot-matrix print is unreliable for OCR). The user can type the date in.
* The health score is a simple, explainable heuristic built on the FSSAI limits and WHO daily guidance, not a clinical or regulatory score.
