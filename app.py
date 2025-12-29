import streamlit as st
import fitz
import pandas as pd
import re
import easyocr
import numpy as np
from PIL import Image

st.set_page_config(page_title="CNH OCR Profissional", layout="wide")
st.title("Extração de Dados e Filiação - CNH")

@st.cache_resource
def get_reader():
    return easyocr.Reader(["pt"], gpu=False)

def extract_cnh_data(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    reader = get_reader()
    # detail=1 retorna as coordenadas para sabermos quem está abaixo de quem
    results = reader.readtext(np.array(img), detail=1)
    
    full_text = " ".join([res[1] for res in results])
    
    data = {}
    
    # 1. CPF e DATAS (Regex Direto)
    cpf_match = re.search(r'(\d{3}[\s.,-]?\d{3}[\s.,-]?\d{3}[\s.,-]?\d{2})', full_text)
    data['CPF'] = cpf_match.group(1) if cpf_match else "Não encontrado"
    
    datas = re.findall(r'(\d{2}/\d{2}/\d{4})', full_text)
    data['Data Nascimento'] = datas[0] if len(datas) > 0 else ""
    data['Validade'] = datas[1] if len(datas) > 1 else ""
    
    # 2. Lógica para NOME e FILIAÇÃO (Baseado em proximidade)
    # Procuramos o índice do texto "NOME" e pegamos o que vem logo depois
    nome = "Não encontrado"
    filiacao = []
    
    for i, res in enumerate(results):
        txt = res[1].upper()
        
        # O nome do portador geralmente é o primeiro bloco grande de texto após a palavra "NOME"
        if "NOME" == txt or "NOMF" in txt:
            if i + 1 < len(results):
                nome = results[i+1][1].strip()
        
        # A filiação vem após "FILIAÇÃO" e geralmente ocupa duas linhas
        if "FILIA" in txt or "FILIAÇÃO" in txt:
            if i + 1 < len(results):
                filiacao.append(results[i+1][1].strip())
            if i + 2 < len(results):
                # Verifica se a segunda linha também parece um nome (sem números)
                if not any(char.isdigit() for char in results[i+2][1]):
                    filiacao.append(results[i+2][1].strip())

    data['Nome'] = nome
    data['Filiação'] = " / ".join(filiacao) if filiacao else "Não encontrada"

    # 3. Registro (11 dígitos isolados)
    reg_match = re.search(r'\b(\d{11})\b', full_text)
    data['Nº Registro'] = reg_match.group(1) if reg_match else ""

    return data, img

# --- Interface Streamlit ---
file = st.file_uploader("Upload CNH PDF", type="pdf")

if file:
    with st.spinner("Extraindo dados com inteligência de contexto..."):
        dados, img_doc = extract_cnh_data(file.getvalue())
        
        st.image(img_doc, use_container_width=True)
        
        df = pd.DataFrame([dados])
        
        st.subheader("Tabela de Dados Extraídos")
        st.dataframe(df, use_container_width=True)
        
        st.download_button("Exportar CSV", df.to_csv(index=False).encode('utf-8'), "cnh_extraida.csv")