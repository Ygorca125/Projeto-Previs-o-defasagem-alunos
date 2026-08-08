import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos - Risco de Defasagem",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Predição de Risco de Defasagem")
st.markdown("### Passos Mágicos")
st.write("Escolha o tipo de aluno e insira os indicadores para estimar a probabilidade de risco.")

# Carregar os dois modelos
@st.cache_resource
def carregar_modelos():
    modelo_sem = joblib.load("modelo_sem_delta.pkl")
    modelo_com = joblib.load("modelo_com_delta.pkl")
    return modelo_sem, modelo_com

try:
    modelo_sem, modelo_com = carregar_modelos()
except Exception as e:
    st.error(f"Erro ao carregar os modelos: {e}")
    st.stop()

# Escolha do tipo de aluno
st.subheader("1. Tipo de Aluno")
tipo_aluno = st.radio(
    "O aluno tem pelo menos 2 anos de histórico no programa?",
    ["Não — Aluno novo (sem histórico)", "Sim — Aluno com histórico (2 anos ou mais)"],
    index=0
)

# Inputs comuns
st.subheader("2. Indicadores do Aluno (Ano Atual)")

col1, col2 = st.columns(2)

with col1:
    ida = st.number_input("IDA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    ieg = st.number_input("IEG", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    iaa = st.number_input("IAA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    ips = st.number_input("IPS", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

with col2:
    ipp = st.number_input("IPP", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    ipv = st.number_input("IPV", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    fase_num = st.number_input("Fase (número)", min_value=0, max_value=8, value=3, step=1)

# Inputs extras (só se tiver histórico)
if tipo_aluno.startswith("Sim"):
    st.subheader("3. Variação em relação ao ano anterior (Deltas)")
    st.caption("Informe quanto o indicador mudou em relação ao ano passado (pode ser negativo).")

    col3, col4 = st.columns(2)
    with col3:
        delta_ida = st.number_input("Delta IDA", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        delta_ieg = st.number_input("Delta IEG", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        delta_iaa = st.number_input("Delta IAA", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    with col4:
        delta_ips = st.number_input("Delta IPS", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        delta_ipp = st.number_input("Delta IPP", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        delta_ipv = st.number_input("Delta IPV", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)


# Botão de predição
if st.button("Calcular Probabilidade de Risco", type="primary"):

    if tipo_aluno.startswith("Não"):
        # ----- Modelo SEM delta -----
        dados = pd.DataFrame([[
            ida, ieg, iaa, ips, ipp, ipv, fase_num
        ]], columns=modelo_sem["features"])

        probabilidade = modelo_sem["modelo"].predict_proba(dados)[0][1]
        threshold = modelo_sem["threshold"]
        modelo_usado = "Sem Delta (Aluno Novo)"

    else:
        # ----- Modelo COM delta -----
        dados = pd.DataFrame([[
            ida, ieg, iaa, ips, ipp, ipv, fase_num,
            delta_ida, delta_ieg, delta_iaa, delta_ips, delta_ipp, delta_ipv
        ]], columns=modelo_com["features"])

        probabilidade = modelo_com["modelo"].predict_proba(dados)[0][1]
        threshold = modelo_com["threshold"]
        modelo_usado = "Com Delta (Aluno com Histórico)"

    # Classificação
    classificacao = "Em Risco" if probabilidade >= threshold else "Sem Risco"


    # Resultado
    st.markdown("---")
    st.subheader("Resultado da Predição")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Probabilidade de Risco", f"{probabilidade:.1%}")

    with col_b:
        if classificacao == "Em Risco":
            st.error(f"Classificação: {classificacao}")
        else:
            st.success(f"Classificação: {classificacao}")

    st.progress(float(probabilidade))
    st.caption(f"Modelo utilizado: **{modelo_usado}** | Threshold: {threshold:.0%}")
