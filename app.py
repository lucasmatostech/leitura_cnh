import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import re
import easyocr
import numpy as np
from PIL import Image

# Configuração da página
st.set_page_config(page_title="Leitor CNH Oficial", layout="wide")
st.title("Extração Inteligente de Dados CNH")

# Cache para carregar o modelo apenas uma vez (evita erro de memória na nuvem)
@st.cache_resource
def load_reader():
    # 'gpu=False' é necessário para o Streamlit Cloud gratuito
    return easyocr.Reader(['pt'], gpu=False)

def extrair_dados_cnh(pdf_bytes):
    # 1. Converte PDF para imagem com alta resolução (Matrix 4x4)
    # Isso resolve erros como 'SANIOS' em vez de 'SANTOS'
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4)) 
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)
    
    # 2. Executa o OCR
    reader = load_reader()
    resultados = reader.readtext(img_np, detail=0)
    
    # Padronização do texto para busca
    texto_limpo = [t.strip().upper() for t in resultados if len(t.strip()) > 1]
    texto_full = " ".join(texto_limpo)
    
    dados = {
        "Nome": "Não encontrado",
        "CPF": "Não encontrado",
        "Nascimento": "Não encontrado",
        "Validade": "Não encontrada",
        "Filiação": "Não encontrada"
    }

    filiacao_lista = []
    
    # 3. Lógica de extração baseada em âncoras
    for i, t in enumerate(texto_limpo):
        # Captura o NOME (que vem logo após a palavra "NOME")
        if "NOME" == t and i + 1 < len(texto_limpo):
            nome_extraido = texto_limpo[i+1]
            # Correções automáticas para falhas comuns de OCR
            dados["Nome"] = nome_extraido.replace("SANIOS", "SANTOS").replace("AMORIN", "AMORIM").replace("NATOS", "MATOS")

        # Captura a FILIAÇÃO (pega as próximas duas linhas após "FILIAÇÃO")
        if "FILIA" in t:
            # Tenta pegar até 3 linhas seguintes para garantir os dois nomes
            for j in range(1, 4):
                if i + j < len(texto_limpo):
                    candidato = texto_limpo[i+j]
                    # Para se encontrar números (CPF/Registro) ou rótulos de data
                    if any(char.isdigit() for char in candidato) or "DATA" in candidato:
                        break
                    filiacao_lista.append(candidato.replace("NATOS", "MATOS"))

    if filiacao_lista:
        dados["Filiação"] = " / ".join(filiacao_lista)

    # 4. Regex para campos de formato fixo (CPF e Datas)
    cpf_match = re.search(r'(\d{3}[\s.,-]?\d{3}[\s.,-]?\d{3}[\s.,-]?\d{2})', texto_full)
    if cpf_match: 
        dados["CPF"] = cpf_match.group(1)
    
    datas = re.findall(r'(\d{2}/\d{2}/\d{4})', texto_full)
    if len(datas) >= 1: dados["Nascimento"] = datas[0]
    if len(datas) >= 2: dados["Validade"] = datas[1]

    return dados, img

# --- Interface ---
uploader = st.file_uploader("Faça o upload da CNH (formato PDF)", type="pdf")

if uploader:
    with st.spinner("Analisando documento... Isso pode levar alguns segundos na primeira execução."):
        try:
            res, img_preview = extrair_dados_cnh(uploader.getvalue())
            
            # Exibe a imagem processada
            st.image(img_preview, caption="Documento Analisado", use_container_width=False, width=600)
            
            # Exibe os dados em uma tabela limpa
            st.subheader("Dados Extraídos com Sucesso")
            st.table(pd.DataFrame([res]))
            
            # Botão para baixar os dados
            csv = pd.DataFrame([res]).to_csv(index=False).encode('utf-8')
            st.download_button("Baixar Dados (CSV)", csv, "dados_cnh.csv", "text/csv")
            
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo: {e}")