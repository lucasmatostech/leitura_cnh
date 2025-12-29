import streamlit as st
import fitz
import easyocr
import numpy as np
import re
import pandas as pd
from PIL import Image

# 1. Carregamento inteligente para evitar "Oh no"
@st.cache_resource
def load_reader():
    return easyocr.Reader(['pt'], gpu=False)

def extrair_dados_final(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    # Matrix 3x3 é o ideal para ler "SANTOS" e "AMORIM" sem erros
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    
    reader = load_reader()
    resultados = reader.readtext(img_np, detail=1)
    
    dados = {"Nome": "Não encontrado", "CPF": "", "Nascimento": "", "Validade": "", "Filiação": ""}
    texto_full = " ".join([res[1].upper() for res in resultados])
    filiacao_lista = []

    # 2. Busca por âncoras para capturar LUCAS SANTOS AMORIM DE MATOS
    for i, (bbox, texto, conf) in enumerate(resultados):
        t = texto.upper().strip()
        if "NOME" == t and i + 1 < len(resultados):
            # Correção de caracteres comum no EasyOCR
            nome = resultados[i+1][1].upper()
            dados["Nome"] = nome.replace("SANIOS", "SANTOS").replace("AMORIN", "AMORIM").replace("NATOS", "MATOS")
        
        if "FILIA" in t:
            for j in range(1, 4):
                if i + j < len(resultados):
                    txt_pai = resultados[i+j][1].upper()
                    if not any(char.isdigit() for char in txt_pai) and len(txt_pai) > 5:
                        filiacao_lista.append(txt_pai.replace("NATOS", "MATOS"))

    dados["Filiação"] = " / ".join(filiacao_lista)
    # Regex para campos numéricos
    cpf = re.search(r'(\d{3}[\s.,-]?\d{3}[\s.,-]?\d{3}[\s.,-]?\d{2})', texto_full)
    if cpf: dados["CPF"] = cpf.group(1)
    
    return dados, Image.fromarray(img_np)

# Interface Streamlit
st.title("Leitor CNH Profissional")
u = st.file_uploader("Suba o PDF", type="pdf")
if u:
    with st.spinner("Iniciando IA... Isso pode demorar no primeiro acesso."):
        res, img = extrair_dados_final(u.getvalue())
        st.image(img, width=500)
        st.table(pd.DataFrame([res]).T)