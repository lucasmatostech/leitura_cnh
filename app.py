import streamlit as st
import fitz
import pandas as pd
import re
import easyocr
import numpy as np
from PIL import Image

st.set_page_config(page_title="Leitor CNH Pro", layout="wide")

# Cache para evitar que o app baixe o modelo toda hora e dê erro de timeout
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['pt'], gpu=False)

def extrair_dados(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    reader = load_ocr()
    # detail=1 ajuda a encontrar o texto por proximidade (quem está abaixo de quem)
    results = reader.readtext(np.array(img), detail=1)
    
    dados = {"Nome": "", "Filiação": "", "CPF": "", "Nascimento": "", "Validade": ""}
    texto_completo = " ".join([res[1] for res in results])
    
    # Lógica de proximidade para Nome e Filiação
    filiacao_nomes = []
    for i, res in enumerate(results):
        txt = res[1].upper()
        # Captura o que está imediatamente após a palavra NOME
        if "NOME" == txt and i + 1 < len(results):
            dados["Nome"] = results[i+1][1].strip().upper()
        # Captura as duas linhas após FILIAÇÃO
        if "FILIA" in txt:
            if i + 1 < len(results): filiacao_nomes.append(results[i+1][1].strip().upper())
            if i + 2 < len(results): filiacao_nomes.append(results[i+2][1].strip().upper())

    dados["Filiação"] = " / ".join(filiacao_nomes)
    
    # Regex para os campos padronizados
    cpf_match = re.search(r'(\d{3}[\s.,-]?\d{3}[\s.,-]?\d{3}[\s.,-]?\d{2})', texto_completo)
    dados["CPF"] = cpf_match.group(1) if cpf_match else ""
    
    datas = re.findall(r'(\d{2}/\d{2}/\d{4})', texto_completo)
    if len(datas) >= 1: dados["Nascimento"] = datas[0]
    if len(datas) >= 2: dados["Validade"] = datas[1]

    return dados, img

# Interface
arquivo = st.file_uploader("Suba sua CNH (PDF)", type="pdf")
if arquivo:
    with st.spinner("Processando..."):
        res, img_doc = extrair_dados(arquivo.getvalue())
        st.image(img_doc, use_container_width=True)
        df = pd.DataFrame([res])
        st.subheader("Resultado")
        st.dataframe(df, use_container_width=True)