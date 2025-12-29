import streamlit as st
import fitz
import pandas as pd
import re
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import logging

# Desativa logs do Paddle no terminal para evitar poluição visual
logging.getLogger("ppocr").setLevel(logging.ERROR)

st.set_page_config(page_title="CNH OCR Inteligente", layout="wide")
st.title("Extração de Dados CNH (PaddleOCR)")

@st.cache_resource
def load_ocr_model():
    # Removido o argumento 'show_log' que causava o erro
    return PaddleOCR(use_angle_cls=True, lang='pt')

def extrair_dados_paddle(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    
    ocr = load_ocr_model()
    # No PaddleOCR recente, o log é controlado no método .ocr()
    result = ocr.ocr(img_np, cls=True)
    
    linhas = []
    if result and result[0]:
        for res in result[0]:
            linhas.append(res[1][0])
            
    texto_completo = " ".join(linhas)
    
    dados = {
        "Nome": "Não encontrado",
        "Filiação": "",
        "CPF": "",
        "Nascimento": "",
        "Validade": ""
    }

    # Lógica de Âncoras para capturar os dados do seu PDF 
    filiacao_nomes = []
    for i, linha in enumerate(linhas):
        l_upper = linha.upper()
        
        # Captura o Nome após o rótulo "NOME"
        if "NOME" == l_upper and i + 1 < len(linhas):
            dados["Nome"] = linhas[i+1].upper()
            
        # Captura os nomes dos pais após "FILIAÇÃO" 
        if "FILIA" in l_upper:
            if i + 1 < len(linhas): 
                filiacao_nomes.append(linhas[i+1].upper())
            if i + 2 < len(linhas):
                # Garante que não está pegando uma data ou CPF por engano
                if not any(char.isdigit() for char in linhas[i+2]):
                    filiacao_nomes.append(linhas[i+2].upper())

    dados["Filiação"] = " / ".join(filiacao_nomes)

    # Regex para padrões fixos 
    cpf_match = re.search(r'(\d{3}[\s.,-]?\d{3}[\s.,-]?\d{3}[\s.,-]?\d{2})', texto_completo)
    dados["CPF"] = cpf_match.group(1) if cpf_match else ""
    
    datas = re.findall(r'(\d{2}/\d{2}/\d{4})', texto_completo)
    if len(datas) >= 1: dados["Nascimento"] = datas[0]
    if len(datas) >= 2: dados["Validade"] = datas[1]

    return dados, Image.fromarray(img_np)

uploader = st.file_uploader("Envie o PDF da CNH", type="pdf")
if uploader:
    with st.spinner("Analisando documento..."):
        res, img_view = extrair_dados_paddle(uploader.getvalue())
        st.image(img_view, use_container_width=True)
        st.subheader("Dados Extraídos")
        st.dataframe(pd.DataFrame([res]), use_container_width=True)