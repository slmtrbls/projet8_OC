import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Désactiver oneDNN pour des résultats numériques plus stables (facultatif)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from PIL import Image

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse

import logging

# Tentative d'import Keras (Keras 3) avec repli vers tf.keras
try:
	from keras.models import load_model  # type: ignore
	KERAS_BACKEND = "keras"
except Exception:  # pragma: no cover
	try:
		from tensorflow.keras.models import load_model  # type: ignore
		KERAS_BACKEND = "tf.keras"
	except Exception as exc:  # pragma: no cover
		load_model = None  # type: ignore
		KERAS_BACKEND = "unavailable"

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR = BASE_DIR / "masks"
MODEL_PATH = BASE_DIR / "model" / "UNet_Lite_AUG_best.keras"

logger = logging.getLogger("segmentation_api")
if not logger.handlers:
	logging.basicConfig(level=logging.INFO)

# Palette 8 classes: 0 Vide, 1 Personne, 2 Véhicule, 3 Construction,
# 4 Objet, 5 Nature, 6 Ciel, 7 Route/Sol
EIGHT_CLASS_PALETTE: List[Tuple[int, int, int]] = [
	(0, 0, 0),        # 0 Vide -> noir
	(220, 20, 60),    # 1 Personne -> rouge
	(0, 0, 142),      # 2 Véhicule -> bleu foncé
	(70, 70, 70),     # 3 Construction -> gris bâtiment
	(250, 170, 30),   # 4 Objet -> orange
	(107, 142, 35),   # 5 Nature -> vert
	(70, 130, 180),   # 6 Ciel -> bleu ciel
	(128, 64, 128),   # 7 Route/Sol -> violet route
]

app = FastAPI(title="Segmentation API", version="1.0.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

_model = None
_model_input_hw: Optional[Tuple[int, int]] = None


def _list_ids() -> List[str]:
	if not IMAGES_DIR.exists():
		return []
	ids: List[str] = []
	for name in os.listdir(IMAGES_DIR):
		if name.endswith("_leftImg8bit.png"):
			ids.append(name.replace("_leftImg8bit.png", ""))
	return sorted(ids)


def _image_path_for(image_id: str) -> Path:
	return IMAGES_DIR / f"{image_id}_leftImg8bit.png"


def _mask_color_path_for(image_id: str) -> Path:
	return MASKS_DIR / f"{image_id}_gtFine_color.png"


def _load_model_once():
	global _model, _model_input_hw
	if _model is not None:
		return
	if load_model is None:
		raise RuntimeError("Keras indisponible. Installez keras>=3 ou tensorflow.")
	if not MODEL_PATH.exists():
		raise RuntimeError(f"Modèle introuvable: {MODEL_PATH}")
	# Essayez avec compile=False (évite les dépendances de compilation/custom_objects)
	try:
		_model_local = load_model(str(MODEL_PATH), compile=False)  # type: ignore[arg-type]
	except TypeError:
		# Anciennes versions sans compile kw
		_model_local = load_model(str(MODEL_PATH))  # type: ignore[call-arg]
	except Exception as exc:
		logger.exception("Échec load_model(compile=False)")
		# Keras 3: essayer safe_mode=False si nécessaire
		try:
			_model_local = load_model(str(MODEL_PATH), compile=False, safe_mode=False)  # type: ignore[call-arg]
		except Exception as exc2:
			logger.exception("Échec load_model(safe_mode=False)")
			raise
	# Taille d'entrée forcée (H, W) = (512, 1024)
	_model_input_hw = (512, 1024)
	_model = _model_local


def _prepare_input(image: Image.Image, target_hw: Optional[Tuple[int, int]]):
	img = image
	if target_hw is not None:
		h, w = target_hw
		if h > 0 and w > 0 and (img.height != h or img.width != w):
			img = image.resize((w, h), resample=Image.BILINEAR)
	x = np.asarray(img, dtype=np.float32) / 255.0
	if x.ndim == 2:
		x = np.expand_dims(x, axis=-1)
	if x.shape[-1] == 4:
		x = x[:, :, :3]
	x = np.expand_dims(x, axis=0)
	return x, img.size  # (W, H)


def _argmax_labels(pred: np.ndarray) -> np.ndarray:
	if pred.ndim == 4 and pred.shape[-1] > 1:
		labels = np.argmax(pred, axis=-1)[0]
	elif pred.ndim == 4 and pred.shape[-1] == 1:
		labels = np.squeeze(pred, axis=(0, -1))
	elif pred.ndim == 3:
		labels = np.argmax(pred, axis=-1)
	else:
		raise ValueError("Forme de sortie du modèle non supportée")
	return labels.astype(np.uint8)


def _resize_labels_nearest(labels: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
	pil = Image.fromarray(labels, mode="L")
	pil = pil.resize(size_wh, resample=Image.NEAREST)
	return np.array(pil, dtype=np.uint8)


def _colorize(labels: np.ndarray) -> Image.Image:
	h, w = labels.shape[:2]
	palette = np.array(EIGHT_CLASS_PALETTE, dtype=np.uint8)
	max_id = int(labels.max()) if labels.size else 0
	num_colors = palette.shape[0]
	if max_id >= num_colors:
		labels = np.clip(labels, 0, num_colors - 1)
	rgb = palette[labels]
	return Image.fromarray(rgb, mode="RGB")


@app.get("/")
async def root():
	return {"status": "ok", "keras": KERAS_BACKEND}


@app.get("/ids")
async def list_ids() -> JSONResponse:
	ids = _list_ids()
	return JSONResponse(ids)


@app.get("/image/{image_id}")
async def get_image(image_id: str):
	path = _image_path_for(image_id)
	if not path.exists():
		raise HTTPException(status_code=404, detail="Image introuvable")
	return FileResponse(path)


@app.get("/mask/original/{image_id}")
async def get_original_mask(image_id: str):
	path = _mask_color_path_for(image_id)
	if not path.exists():
		raise HTTPException(status_code=404, detail="Mask original introuvable")
	return FileResponse(path)


@app.get("/predict/{image_id}")
async def predict_mask(image_id: str):
	try:
		_load_model_once()
	except Exception as exc:
		logger.exception("Erreur lors du chargement du modèle")
		raise HTTPException(status_code=503, detail=f"Erreur chargement modèle: {exc}")

	img_path = _image_path_for(image_id)
	if not img_path.exists():
		raise HTTPException(status_code=404, detail="Image introuvable")

	image = Image.open(img_path).convert("RGB")
	original_size_wh = (image.width, image.height)

	try:
		x, _ = _prepare_input(image, _model_input_hw)
		pred = _model.predict(x)  # type: ignore
		labels = _argmax_labels(pred)
		if labels.shape != (image.height, image.width):
			labels = _resize_labels_nearest(labels, original_size_wh)
	except Exception:
		# Repli vers une taille par défaut si la prédiction échoue
		x, _ = _prepare_input(image, (512, 1024))
		pred = _model.predict(x)  # type: ignore
		labels = _argmax_labels(pred)
		if labels.shape != (image.height, image.width):
			labels = _resize_labels_nearest(labels, original_size_wh)

	colored = _colorize(labels)
	buf = io.BytesIO()
	colored.save(buf, format="PNG")
	buf.seek(0)
	return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
	import uvicorn
	uvicorn.run("app.backend:app", host="0.0.0.0", port=8000, reload=True)
