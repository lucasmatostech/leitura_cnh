import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import pandas as pd
import re
import unicodedata
import traceback

st.set_page_config(page_title="CNH - OCR (EasyOCR) Frente", layout="wide")
st.title("CNH (PDF) → OCR (EasyOCR) → DataFrame (Frente)")
st.caption("Dica: para OCR ficar bom, use PDF/scan nítido. Processamento em memória.")

MAX_MB = 8

# -------------------------
# Cache do EasyOCR
# -------------------------
@st.cache_resource
def get_reader():
    import easyocr
    # pt + en ajuda quando OCR confunde acentos/abreviações
    return easyocr.Reader(["pt", "en"], gpu=False)

# -------------------------
# PDF -> imagem (página 1)
# -------------------------
def pdf_page1_to_image(pdf_bytes: bytes, zoom: float = 2.0) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

# -------------------------
# Pré-processamento simples
# -------------------------
def preprocess(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    # binarização leve (melhora texto impresso)
    bw = g.point(lambda x: 255 if x > 165 else 0, mode="1")
    return bw.convert("L")

# -------------------------
# Recorte: "Frente da CNH"
# (metade esquerda, removendo margens)
# -------------------------
def crop_cnh_front(page_img: Image.Image) -> Image.Image:
    w, h = page_img.size

    # Ajuste fino: pega a metade esquerda e corta bordas.
    # (funciona na maioria dos PDFs de CNH onde o QR fica à direita)
    x0 = int(0.05 * w)
    y0 = int(0.12 * h)
    x1 = int(0.58 * w)   # corta antes do QR (direita)
    y1 = int(0.86 * h)

    return page_img.crop((x0, y0, x1, y1))

# -------------------------
# Normalização de texto
# -------------------------
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def norm_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u00a0", " ")
    s = strip_accents(s).upper()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")

# -------------------------
# OCR
# -------------------------
def ocr_image(reader, img: Image.Image) -> str:
    arr = np.array(img)
    lines = reader.readtext(arr, detail=0, paragraph=True)
    return "\n".join([l.strip() for l in lines if l and l.strip()])

# -------------------------
# Parser (focado em CNH frente)
# -------------------------
def parse_front_fields(ocr_raw: str) -> dict:
    t = norm_text(ocr_raw)
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]

    # CPF (11 dígitos)
    cpf = None
    mcpf = re.search(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2})\b", t)
    if mcpf:
        cpf = digits_only(mcpf.group(1))
    else:
        m11 = re.search(r"\b(\d{11})\b", digits_only(t))
        cpf = m11.group(1) if m11 else None

    # Datas dd/mm/aaaa (vamos procurar próximas de palavras, mas com OCR zoado aceitamos fallback)
    dates = re.findall(r"\b(\d{2}/\d{2}/\d{4})\b", t)
    dt_nasc = None
    dt_valid = None
    dt_emiss = None
    dt_1hab = None

    # Heurísticas simples por label (tolerante a OCR)
    joined = " | ".join(lines)

    def pick_date_near(label_keywords, max_dist=140):
        # acha posição do label no texto "joined"
        label_pos = None
        for kw in label_keywords:
            p = joined.find(kw)
            if p != -1:
                label_pos = p
                break
        if label_pos is None:
            return None
        best = None
        best_dist = 10**9
        for d in dates:
            dp = joined.find(d)
            if dp != -1:
                dist = abs(dp - label_pos)
                if dist < best_dist and dist <= max_dist:
                    best_dist = dist
                    best = d
        return best

    dt_nasc = pick_date_near(["NASC", "NASCIMENTO", "DATA NASC"])
    dt_valid = pick_date_near(["VALID", "VALIDADE"])
    dt_emiss = pick_date_near(["EMISS", "EMISSAO"])
    dt_1hab  = pick_date_near(["HABILIT", "1A HAB", "1 HAB", "HABILITAC"])

    # Registro CNH (9 a 12 dígitos)
    registro = None
    for ln in lines:
        if "REGIST" in ln:
            cand = re.findall(r"\d{9,12}", digits_only(ln))
            if cand:
                registro = cand[0]
                break

    # Categoria (A/B/AB/AC etc) - CNH BR normalmente A, B, AB, AC, AD, AE
    categoria = None
    for ln in lines:
        if "CAT" in ln:
            m = re.search(r"\b(AE|AD|AC|AB|A|B|C|D|E)\b", ln)
            if m:
                categoria = m.group(1)
                break

    # Nome: pega a “melhor linha” com letras (evita cabeçalho)
    blacklist = ["REPUBLICA", "FEDERATIVA", "BRASIL", "MINISTERIO", "SECRETARIA", "SENATRAN", "DENATRAN", "SERPRO", "QRCODE", "QR-CODE"]
    name_candidates = []
    for ln in lines:
        if any(b in ln for b in blacklist):
            continue
        if len(ln) < 10:
            continue
        if len(digits_only(ln)) >= 6:
            continue
        # precisa ter pelo menos 2 palavras
        if len(ln.split()) < 2:
            continue
        score = sum(1 for c in ln if c.isalpha()) + len(ln.split()) * 5
        name_candidates.append((score, ln))
    name_candidates.sort(reverse=True)
    nome = name_candidates[0][1] if name_candidates else None

    return {
        "NOME": nome,
        "CPF": cpf,
        "REGISTRO_CNH": registro,
        "CATEGORIA": categoria,
        "DT_NASCIMENTO": dt_nasc,
        "DT_EMISSAO": dt_emiss,
        "DT_VALIDADE": dt_valid,
        "DT_1A_HABILITACAO": dt_1hab,
    }

# -------------------------
# UI
# -------------------------
with st.sidebar:
    st.header("Ajustes")
    zoom = st.slider("Zoom do PDF", 1.4, 3.0, 2.0, 0.2)
    show_debug = st.checkbox("Mostrar texto OCR (debug)", value=True)
    show_crop = st.checkbox("Mostrar recorte da frente", value=True)

pdf_file = st.file_uploader("Envie a CNH (PDF)", type=["pdf"])

if not pdf_file:
    st.info("Envie um PDF para começar.")
    st.stop()

pdf_bytes = pdf_file.getvalue()
size_mb = len(pdf_bytes) / (1024 * 1024)
if size_mb > MAX_MB:
    st.error(f"Arquivo muito grande ({size_mb:.2f} MB). Limite: {MAX_MB} MB.")
    st.stop()

run = st.button("Executar OCR na FRENTE da CNH", type="primary", use_container_width=True)

try:
    page_img = pdf_page1_to_image(pdf_bytes, zoom=zoom)
    front = crop_cnh_front(page_img)
    front_prep = preprocess(front)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Página 1")
        st.image(page_img, use_container_width=True)

    with col2:
        if show_crop:
            st.subheader("Recorte: Frente da CNH")
            st.image(front, use_container_width=True)

    if not run:
        st.info("Clique no botão para rodar o OCR.")
        st.stop()

    reader = get_reader()

    with st.spinner("Rodando OCR (somente na frente)..."):
        ocr_raw = ocr_image(reader, front_prep)

    if show_debug:
        st.subheader("Texto OCR (frente) - debug")
        st.text_area("OCR", ocr_raw, height=220)

    fields = parse_front_fields(ocr_raw)
    df = pd.DataFrame([fields])

    st.subheader("DataFrame extraído (frente)")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Baixar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cnh_frente_extraida.csv",
        mime="text/csv",
        use_container_width=True
    )

except Exception:
    st.error("Erro no processamento.")
    st.code(traceback.format_exc())
