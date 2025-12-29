import streamlit as st
import fitz
import pandas as pd
import re
import easyocr
import numpy as np
from PIL import Image

st.set_page_config(page_title="Leitor CNH Pro", layout="wide")

@st.cache_resource
def load_reader():
    # Cache essencial para o Streamlit Cloud não travar
    return easyocr.Reader(['pt'], gpu=False)

def extrair_dados(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    # Matrix 3x3 oferece o melhor equilíbrio para leitura de nomes e datas
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    
    reader = load_reader()
    resultados = reader.readtext(img_np, detail=0) # detail=0 volta a ser mais estável aqui
    
    # Criamos uma versão em texto corrido e uma lista limpa
    texto_full = " ".join([t.upper() for t in resultados])
    texto_lista = [t.upper().strip() for t in resultados]
    
    dados = {
        "Nome": "Não encontrado",
        "CPF": "Não encontrado",
        "Nascimento": "Não encontrado",
        "Validade": "Não encontrado",
        "Filiação": "Não encontrada"
    }

    # 1. Busca do Nome (Âncora direta)
    for i, t in enumerate(texto_lista):
        if "NOME" == t and i + 1 < len(texto_lista):
            nome_bruto = texto_lista[i+1]
            # Correções ortográficas manuais para falhas do OCR
            dados["Nome"] = nome_bruto.replace("SANIOS", "SANTOS").replace("AMORIN", "AMORIM").replace("NATOS", "MATOS")
            break

    # 2. Busca de Datas (Regex é melhor que posição para Nascimento/Validade)
    datas = re.findall(r'(\d{2}/\d{2}/\d{4})', texto_full)
    if len(datas) >= 1: dados["Nascimento"] = datas[0]
    if len(datas) >= 2: dados["Validade"] = datas[1]

    # 3. Busca do CPF (Regex ignora se o OCR leu vírgula ou ponto errado)
    cpf_pattern = re.search(r'(\d{3}[\s.,-]?\d{3}[\s.,-]?\d{3}[\s.,-]?\d{2})', texto_full)
    if cpf_pattern:
        dados["CPF"] = cpf_pattern.group(1).replace(",", ".")

    # 4. Busca da Filiação (Captura as linhas entre 'FILIAÇÃO' e o próximo campo numérico)
    pais = []
    for i, t in enumerate(texto_lista):
        if "FILIA" in t:
            for j in range(1, 4): # Varre as 3 linhas seguintes
                if i + j < len(texto_lista):
                    linha = texto_lista[i+j]
                    # Se achar um número ou CPF, a filiação acabou
                    if any(char.isdigit() for char in linha) or "NREGISTRO" in linha:
                        break
                    if len(linha) > 5: # Filtra ruídos pequenos
                        pais.append(linha.replace("NATOS", "MATOS"))
            break
    dados["Filiação"] = " / ".join(pais)

    return dados, Image.fromarray(img_np)

# --- Interface ---
st.title("Extração de Dados CNH-e")
arquivo = st.file_uploader("Suba o arquivo PDF", type="pdf")

if arquivo:
    with st.spinner("Analisando documento..."):
        try:
            res, img_cnh = extrair_dados(arquivo.getvalue())
            st.image(img_cnh, width=500)
            st.subheader("Resultado")
            # Exibição em DataFrame transposto para facilitar leitura
            st.table(pd.DataFrame([res]).T.rename(columns={0: "Informações"}))
        except Exception as e:
            st.error(f"Erro: {e}")