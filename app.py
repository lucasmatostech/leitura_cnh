import streamlit as st
import fitz
import pandas as pd
import re
import easyocr
import numpy as np
from PIL import Image

st.set_page_config(page_title="Leitor CNH Oficial", layout="wide")

# Cache para carregar o modelo apenas uma vez e não dar "Oh no" no Streamlit
@st.cache_resource
def load_reader():
    # 'gpu=False' é essencial para o Streamlit Cloud gratuito
    return easyocr.Reader(['pt'], gpu=False)

def extrair_dados(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)
    
    reader = load_reader()
    # detail=0 simplifica o retorno para apenas uma lista de textos
    resultados = reader.readtext(img_np, detail=0)
    texto_completo = " ".join(resultados)
    
    dados = {
        "Nome": "Não encontrado",
        "CPF": "Não encontrado",
        "Data Nascimento": "Não encontrada",
        "Validade": "Não encontrada",
        "Filiação": "Não encontrada"
    }

    # Lógica de busca inteligente por palavras-chave
    pais = []
    for i, txt in enumerate(resultados):
        t = txt.upper()
        # Se achar "NOME", o seu nome está logo abaixo [cite: 9, 10]
        if "NOME" == t and i + 1 < len(resultados):
            dados["Nome"] = resultados[i+1].upper()
        
        # Se achar "FILIAÇÃO", captura as próximas linhas [cite: 15, 18, 19]
        if "FILIA" in t:
            if i + 1 < len(resultados): pais.append(resultados[i+1].upper())
            if i + 2 < len(resultados):
                # Só adiciona a segunda linha se não for data/número
                if not any(char.isdigit() for char in resultados[i+2]):
                    pais.append(resultados[i+2].upper())

    dados["Filiação"] = " / ".join(pais)

    # Regex para padrões fixos (CPF e Datas) [cite: 14, 17, 25]
    cpf = re.search(r'(\d{3}[\s.,-]?\d{3}[\s.,-]?\d{3}[\s.,-]?\d{2})', texto_completo)
    if cpf: dados["CPF"] = cpf.group(1)
    
    datas = re.findall(r'(\d{2}/\d{2}/\d{4})', texto_completo)
    if len(datas) >= 1: dados["Data Nascimento"] = datas[0]
    if len(datas) >= 2: dados["Validade"] = datas[1]

    return dados, img

# Interface do Usuário
st.title("Extração de Dados CNH-e")
arquivo = st.file_uploader("Suba o PDF da sua CNH", type="pdf")

if arquivo:
    with st.spinner("Processando... Na primeira execução pode demorar 1 minuto."):
        res, img_cnh = extrair_dados(arquivo.getvalue())
        st.image(img_cnh, caption="Visualização do Documento", width=400)
        st.subheader("Dados Extraídos")
        st.table(pd.DataFrame([res]))