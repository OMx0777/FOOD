"""FoodGuard AI - Flask web server.

    python app.py            -> http://127.0.0.1:5000

Endpoints
    GET  /              web interface
    POST /api/analyze   multipart form: images (1-3 files), category, product_name?, barcode?,
                        expiry_date? (YYYY-MM-DD, optional)
    GET  /api/health    model / device status
"""
import logging
import threading
from datetime import date

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from foodguard import config, models, pipeline

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_IMAGES = 3

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
log = logging.getLogger("foodguard")

# EasyOCR and the CUDA models are not guaranteed thread-safe: one analysis at a time.
_analysis_lock = threading.Lock()


def _load_image(file) -> Image.Image:
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        raise ValueError(f"'{file.filename}' is not a supported image type ({', '.join(sorted(ALLOWED_EXT))}).")
    try:
        img = Image.open(file.stream)
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError(f"'{file.filename}' could not be read as an image.")
    img = ImageOps.exif_transpose(img)          # phone photos carry rotation in EXIF
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    return img.convert("RGB")


@app.get("/")
def index():
    return render_template("index.html", categories=config.CATEGORIES,
                           supported=[p.replace("_", " ") for p in config.SUPPORTED_PRODUCE])


@app.post("/api/analyze")
def analyze():
    files = [f for f in request.files.getlist("images") if f and f.filename]
    category = request.form.get("category", "")
    if not files:
        return jsonify(error="Please upload at least one image."), 400
    if len(files) > MAX_IMAGES:
        return jsonify(error=f"Upload at most {MAX_IMAGES} images."), 400
    if category not in config.CATEGORIES:
        return jsonify(error="Please choose a food category."), 400
    try:
        images = [_load_image(f) for f in files]
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    expiry_raw = (request.form.get("expiry_date") or "").strip()
    try:
        expiry_date = date.fromisoformat(expiry_raw) if expiry_raw else None
    except ValueError:
        return jsonify(error="Expiry date must be a valid date (YYYY-MM-DD)."), 400

    with _analysis_lock:
        try:
            result = pipeline.analyse(images, category,
                                      product_name=request.form.get("product_name"),
                                      barcode=request.form.get("barcode"),
                                      expiry_date=expiry_date)
        except Exception:  # noqa: BLE001 - report any model failure to the client
            log.exception("Analysis failed")
            return jsonify(error="Analysis failed on the server - see the console log."), 500
    return jsonify(result)


@app.get("/api/health")
def health():
    return jsonify(models.status())


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error="Upload too large (max 30 MB in total)."), 413


def _warm_up():
    """Load all models in the background so the first request is fast."""
    try:
        models.get_clip()
        models.get_ocr_reader()
        models.get_spoilage_model()
        log.info("All models loaded on %s", models.DEVICE)
    except models.ModelNotTrainedError as exc:
        log.warning("%s", exc)
    except Exception:  # noqa: BLE001
        log.exception("Model warm-up failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(" * FoodGuard AI: open http://127.0.0.1:5000  (models load in the background)")
    threading.Thread(target=_warm_up, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
