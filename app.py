import streamlit as st
import base64

st.set_page_config(page_title="Visualização de CNH", layout="wide")

st.title("Upload e Visualização de CNH (PDF)")

uploaded = st.file_uploader("Envie sua CNH (PDF)", type=["pdf"])

if uploaded:
    pdf_bytes = uploaded.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    st.success("Arquivo carregado com sucesso!")

    # HTML com PDF.js embutido
    pdf_viewer = f"""
    <iframe
        src="https://mozilla.github.io/pdf.js/web/viewer.html?file=data:application/pdf;base64,{b64_pdf}"
        width="100%"
        height="800px"
        style="border:none;">
    </iframe>
    """

    st.components.v1.html(pdf_viewer, height=820)

    st.download_button(
        "Baixar PDF",
        data=pdf_bytes,
        file_name=uploaded.name,
        mime="application/pdf"
    )
