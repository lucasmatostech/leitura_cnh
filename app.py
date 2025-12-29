import streamlit as st
import fitz
import pytesseract
import pandas as pd
import re
from PIL import Image
import numpy as np

st.set_page_config(page_title="CNH OCR Leve", layout="wide")

def extrair_dados_leve(pdf_bytes):
    # 1. Converte PDF para imagem
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 2. OCR usando Tesseract (muito mais leve que EasyOCR)
    # config: '--psm 6' assume um bloco de texto uniforme
    texto = pytesseract.image_to_string(img, lang='por', config='--psm 6')
    
    # 3. Mineração de Dados (Lógica de busca por âncoras e padrões)
    linhas = [l.strip() for l in texto.split('\n') if l.strip()]
    dados = {"Nome": "", "Filiação": "", "CPF": "", "Nascimento": "", "Validade": ""}

    for i, linha in enumerate(linhas):
        # Nome: geralmente a linha seguinte ao título "NOME"
        if "NOME" in linha.upper() and i + 1 < len(linhas):
            dados["Nome"] = linhas[i+1].upper()
        
        # Filiação: busca as linhas após o título
        if "FILIA" in linha.upper():
            filiacao = []
            if i + 1 < len(linhas): filiacao.append(linhas[i+1].upper())
            if i + 2 < len(linhas): 
                if not any(char.isdigit() for char in linhas[i+2]):
                    filiacao.append(linhas[i+2].upper())
            dados["Filiação"] = " / ".join(filiacao)

    # Regex para CPF e Datas
    cpf = re.search(r'(\d{3}[\s.,-]?\d{3}[\s.,-]?\d{3}[\s.,-]?\d{2})', texto)
    dados["CPF"] = cpf.group(1) if cpf else ""
    
    datas = re.findall(r'(\d{2}/\d{2}/\d{4})', texto)
    if len(datas) >= 1: dados["Nascimento"] = datas[0]
    if len(datas) >= 2: dados["Validade"] = datas[1]

    return dados, img

# Interface
uploader = st.file_uploader("Upload CNH (PDF)", type="pdf")
if uploader:
    res, img_preview = extrair_dados_leve(uploader.getvalue())
    st.image(img_preview, use_container_width=True)
    st.dataframe(pd.DataFrame([res]), use_container_width=True)