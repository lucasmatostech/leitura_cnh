import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import pandas as pd
import re
import traceback

st.set_page_config(page_title="CNH - OCR (EasyOCR) Layout Clássico", layout="wide")
st.title("CNH (PDF) → OCR (EasyOCR) → DataFrame (layout clássico)")

MAX_MB = 8

# -------------------------
# EasyOCR cache
# -------------------------
@st.cache_resource
def get_reader():
    import easyocr
    return easyocr.Reader(["pt", "en"], gpu=False)

# -------------------------
# PDF -> imagem (página 1)
# -------------------------
def pdf_page_to_image(pdf_bytes: bytes, page_index: int = 0, zoom: float = 2.2) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

# -------------------------
# Crop absoluto por sliders (0..1)
# -------------------------
def crop_abs_rel(img: Image.Image, left, top, right, bottom):
    w, h = img.size
    x0 = int(left * w)
    y0 = int(top * h)
    x1 = int(right * w)
    y1 = int(bottom * h)
    # garante mínimo
    x0 = max(0, min(x0, w - 2))
    y0 = max(0, min(y0, h - 2))
    x1 = max(x0 + 2, min(x1, w))
    y1 = max(y0 + 2, min(y1, h))
    return img.crop((x0, y0, x1, y1))

# -------------------------
# Pré-processamento leve (NÃO binariza forte pra não “virar QR”)
# -------------------------
def prep_for_ocr(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    return g

# -------------------------
# Crop relativo dentro da CNH
# -------------------------
def crop_rel(img: Image.Image, box_rel):
    w, h = img.size
    x0, y0, x1, y1 = box_rel
    return img.crop((int(x0*w), int(y0*h), int(x1*w), int(y1*h)))

# -------------------------
# OCR helper
# -------------------------
def ocr_field(reader, img: Image.Image) -> str:
    arr = np.array(img)
    lines = reader.readtext(arr, detail=0, paragraph=True)
    txt = " ".join([l.strip() for l in lines if l and l.strip()])
    return re.sub(r"\s{2,}", " ", txt).strip()

def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def find_date(text: str):
    m = re.search(r"(\d{2}/\d{2}/\d{4})", text or "")
    return m.group(1) if m else None

def find_cat(text: str):
    t = (text or "").upper()
    m = re.search(r"\b(AE|AD|AC|AB|A|B|C|D|E)\b", t)
    return m.group(1) if m else None

# -------------------------
# ROIs (layout clássico) DENTRO do recorte da frente da CNH
# Importante: esses valores funcionam quando o recorte-base está “certinho”
# (só a frente, sem QR)
# -------------------------
ROIS = {
    "NOME":       (0.08, 0.10, 0.60, 0.20),
    "DOC_ORG":    (0.60, 0.10, 0.98, 0.16),
    "CPF":        (0.60, 0.16, 0.82, 0.21),
    "DT_NASC":    (0.82, 0.16, 0.98, 0.21),
    "FILIACAO":   (0.60, 0.21, 0.98, 0.34),

    "PERMISSAO":  (0.60, 0.35, 0.74, 0.40),
    "ACC":        (0.74, 0.35, 0.86, 0.40),
    "CATEGORIA":  (0.86, 0.35, 0.98, 0.40),

    "N_REGISTRO": (0.08, 0.41, 0.45, 0.46),
    "VALIDADE":   (0.45, 0.41, 0.70, 0.46),
    "DT_1HAB":    (0.70, 0.41, 0.98, 0.46),

    "OBS":        (0.08, 0.53, 0.98, 0.70),

    "LOCAL":      (0.08, 0.86, 0.72, 0.92),
    "DT_EMISSAO": (0.72, 0.86, 0.98, 0.92),
}

# =========================
# UI
# =========================
with st.sidebar:
    st.header("Configurações")
    zoom = st.slider("Zoom (PDF → imagem)", 1.6, 3.2, 2.4, 0.2)
    mode = st.radio("Modo", ["Calibração do recorte", "Extrair campos"], index=0)
    show_rois = st.checkbox("Mostrar recortes dos campos", value=True)
    show_debug = st.checkbox("Mostrar texto OCR por campo", value=True)

pdf_file = st.file_uploader("Envie a CNH (PDF)", type=["pdf"])

if not pdf_file:
    st.info("Envie um PDF para começar.")
    st.stop()

pdf_bytes = pdf_file.getvalue()
size_mb = len(pdf_bytes) / (1024 * 1024)
if size_mb > MAX_MB:
    st.error(f"Arquivo muito grande ({size_mb:.2f} MB). Limite: {MAX_MB} MB.")
    st.stop()

# Render da página 1
page_img = pdf_page_to_image(pdf_bytes, page_index=0, zoom=zoom)
page_prep = prep_for_ocr(page_img)

# Valores default de recorte (ajuste no seu caso):
# A ideia aqui é você enquadrar SÓ a frente (sem QR).
if "crop" not in st.session_state:
    st.session_state.crop = dict(left=0.04, top=0.10, right=0.62, bottom=0.92)

with st.sidebar:
    st.subheader("Recorte-base (frente da CNH)")
    left = st.slider("Left", 0.0, 1.0, float(st.session_state.crop["left"]), 0.01)
    top = st.slider("Top", 0.0, 1.0, float(st.session_state.crop["top"]), 0.01)
    right = st.slider("Right", 0.0, 1.0, float(st.session_state.crop["right"]), 0.01)
    bottom = st.slider("Bottom", 0.0, 1.0, float(st.session_state.crop["bottom"]), 0.01)

    if st.button("Salvar recorte"):
        st.session_state.crop = dict(left=left, top=top, right=right, bottom=bottom)

# Recorte atual
crop_cfg = st.session_state.crop
front_img = crop_abs_rel(page_img, crop_cfg["left"], crop_cfg["top"], crop_cfg["right"], crop_cfg["bottom"])
front_prep = prep_for_ocr(front_img)

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    st.subheader("Página 1 (render)")
    st.image(page_img, use_container_width=True)

with col2:
    st.subheader("Recorte-base (frente)")
    st.image(front_img, use_container_width=True)
    st.caption("Ajuste os sliders até o recorte mostrar SÓ a frente (sem QR). Depois clique em **Salvar recorte**.")

if mode == "Calibração do recorte":
    st.info("Quando o recorte-base estiver certo, mude o modo para **Extrair campos**.")
    st.stop()

# =========================
# EXTRAÇÃO
# =========================
run = st.button("Executar OCR (por campo)", type="primary", use_container_width=True)

if not run:
    st.stop()

try:
    reader = get_reader()

    raw = {}
    crops = {}

    for k, box in ROIS.items():
        crop = crop_rel(front_prep, box)
        crops[k] = crop
        raw[k] = ocr_field(reader, crop)

    # Pós-processamento
    cpf = digits_only(raw.get("CPF"))
    cpf = cpf if len(cpf) == 11 else None

    n_reg = digits_only(raw.get("N_REGISTRO"))
    n_reg = n_reg if len(n_reg) >= 9 else None

    df = pd.DataFrame([{
        "NOME": raw.get("NOME") or None,
        "DOC_IDENTIDADE_ORG_UF": raw.get("DOC_ORG") or None,
        "CPF": cpf,
        "DT_NASCIMENTO": find_date(raw.get("DT_NASC")),
        "FILIACAO_TEXTO": raw.get("FILIACAO") or None,
        "PERMISSAO": raw.get("PERMISSAO") or None,
        "ACC": raw.get("ACC") or None,
        "CATEGORIA": find_cat(raw.get("CATEGORIA")),
        "N_REGISTRO": n_reg,
        "DT_VALIDADE": find_date(raw.get("VALIDADE")),
        "DT_1A_HABILITACAO": find_date(raw.get("DT_1HAB")),
        "OBSERVACOES": raw.get("OBS") or None,
        "LOCAL": raw.get("LOCAL") or None,
        "DT_EMISSAO": find_date(raw.get("DT_EMISSAO")),
    }])

    st.divider()

    if show_rois:
        st.subheader("Recortes (ROIs) – agora devem cair nos campos (não no QR)")
        cols = st.columns(3)
        for i, (k, img) in enumerate(crops.items()):
            with cols[i % 3]:
                st.image(img, caption=k, use_container_width=True)

    if show_debug:
        st.subheader("Texto OCR por campo (debug)")
        st.json(raw)

    st.subheader("DataFrame extraído")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Baixar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cnh_extraida_layout_classico.csv",
        mime="text/csv",
        use_container_width=True
    )

except Exception:
    st.error("Erro no processamento.")
    st.code(traceback.format_exc())
