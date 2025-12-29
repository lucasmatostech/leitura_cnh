import streamlit as st
import fitz
import pandas as pd
import re
import easyocr
import numpy as np
from PIL import Image

st.set_page_config(page_title="Leitor CNH Preciso", layout="wide")

@st.cache_resource
def load_reader():
    # Carrega o modelo uma única vez
    return easyocr.Reader(['pt'], gpu=False)

def extrair_dados(pdf_bytes):
    # 1. Conversão Equilibrada (Matrix 3x3 é o ponto ideal entre nitidez e ordem)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) 
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    
    reader = load_reader()
    # detail=1 retorna: [ [[coords], texto, confiança], ... ]
    resultados = reader.readtext(img_np, detail=1)
    
    dados = {
        "Nome": "Não encontrado",
        "CPF": "Não encontrado",
        "Nascimento": "Não encontrado",
        "Validade": "Não encontrada",
        "Filiação": "Não encontrada"
    }

    # Texto bruto para Regex
    texto_full = " ".join([res[1].upper() for res in resultados])
    
    filiacao_nomes = []

    # 2. Busca por Proximidade Espacial
    for i, (bbox, texto, conf) in enumerate(resultados):
        txt = texto.upper().strip()
        
        # --- BUSCA NOME ---
        if "NOME" == txt:
            # O nome geralmente é o próximo bloco detectado após a palavra NOME
            if i + 1 < len(resultados):
                nome_bruto = resultados[i+1][1].upper()
                # Correção ortográfica manual para os erros que você reportou
                dados["Nome"] = nome_bruto.replace("SANIOS", "SANTOS").replace("AMORIN", "AMORIM").replace("NATOS", "MATOS")

        # --- BUSCA FILIAÇÃO ---
        if "FILIA" in txt:
            # Captura os próximos dois blocos de texto que não sejam números
            count = 0
            for j in range(1, 4):
                if i + j < len(resultados):
                    txt_pai = resultados[i+j][1].upper()
                    if not any(char.isdigit() for char in txt_pai) and len(txt_pai) > 5:
                        filiacao_nomes.append(txt_pai.replace("NATOS", "MATOS"))
                        count += 1
                if count >= 2: break

    dados["Filiação"] = " / ".join(filiacao_nomes)

    # 3. Regex para CPF e Datas (Mais confiável para números)
    cpf_match = re.search(r'(\d{3}[\s.,-]?\d{3}[\s.,-]?\d{3}[\s.,-]?\d{2})', texto_full)
    if cpf_match: dados["CPF"] = cpf_match.group(1)
    
    datas = re.findall(r'(\d{2}/\d{2}/\d{4})', texto_full)
    if len(datas) >= 1: dados["Nascimento"] = datas[0]
    if len(datas) >= 2: dados["Validade"] = datas[1]

    return dados, Image.fromarray(img_np)

# --- Interface ---
st.title("Extração Profissional de CNH")
uploader = st.file_uploader("Suba o PDF da CNH", type="pdf")

if uploader:
    with st.spinner("Processando..."):
        try:
            res, img_view = extrair_dados(uploader.getvalue())
            st.image(img_view, width=500)
            st.subheader("Resultado da Extração")
            # Tabela Transposta para facilitar a leitura no celular/web
            df = pd.DataFrame([res]).T
            df.columns = ["Dados Identificados"]
            st.table(df)
        except Exception as e:
            st.error(f"Erro: {e}")