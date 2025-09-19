import os
from pathlib import Path
from typing import List

import requests
import streamlit as st

# Configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Segmentation Viewer", layout="wide")
st.title("Segmentation - Démo")

@st.cache_data
def fetch_ids() -> List[str]:
	resp = requests.get(f"{BACKEND_URL}/ids", timeout=10)
	resp.raise_for_status()
	return resp.json() 


def get_image_url(image_id: str) -> str:
	return f"{BACKEND_URL}/image/{image_id}"


def get_original_mask_url(image_id: str) -> str:
	return f"{BACKEND_URL}/mask/original/{image_id}"


def get_predict_mask_url(image_id: str) -> str:
	return f"{BACKEND_URL}/predict/{image_id}"


def _show_image_compat(image, caption: str | None = None) -> None:
	# Compatibilité anciennes versions de Streamlit
	try:
		st.image(image, caption=caption, use_container_width=True)
	except TypeError:
		st.image(image, caption=caption, use_column_width=True)


ids = fetch_ids()
if not ids:
	st.warning("Aucun ID trouvé. Vérifiez le backend et le dossier app/images.")
	st.stop()

selected_id = st.selectbox("Sélectionnez un ID d'image", ids)

cols = st.columns(3)
with cols[0]:
	st.subheader("Image")
	if selected_id:
		_show_image_compat(get_image_url(selected_id))

with cols[1]:
	st.subheader("Mask original")
	if selected_id:
		_show_image_compat(get_original_mask_url(selected_id))

with cols[2]:
	st.subheader("Mask généré")
	generated_placeholder = st.empty()

if st.button("Générer la segmentation", type="primary"):
	with st.spinner("Prédiction en cours..."):
		try:
			pred_url = get_predict_mask_url(selected_id)
			pred_resp = requests.get(pred_url, timeout=120)
			if pred_resp.status_code != 200:
				msg = pred_resp.text or pred_resp.reason
				raise requests.HTTPError(msg, response=pred_resp)
			pred_png_bytes = pred_resp.content
			try:
				generated_placeholder.image(pred_png_bytes, caption="Mask généré", use_container_width=True)
			except TypeError:
				generated_placeholder.image(pred_png_bytes, caption="Mask généré", use_column_width=True)
		except requests.HTTPError as http_exc:
			st.error(f"Erreur backend: {getattr(http_exc.response, 'status_code', '')} - {http_exc}")
		except Exception as exc:
			st.error(f"Erreur pendant la prédiction: {exc}")
