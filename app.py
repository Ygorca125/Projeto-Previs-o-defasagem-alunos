import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Passos Mágicos - Risco de Defasagem",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Predição de Risco de Defasagem")
st.markdown("### Passos Mágicos")
st.write("Insira os indicadores do aluno (ano base + ano seguinte) para estimar a probabilidade de risco.")


@st.cache_resource
def carregar_modelo():
    artefatos = joblib.load('modelo_rf_deltas_risco.pkl')
    return artefatos

artefatos = carregar_modelo()
modelo = artefatos['modelo']
features = artefatos['features']
threshold = artefatos['threshold']

st.subheader("Indicadores do Aluno")

st.markdown("#### Ano Base (ex: 2020)")
col1, col2 = st.columns(2)

with col1:
    inde_2020 = st.number_input("INDE (Ano Base)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ida_2020 = st.number_input("IDA (Ano Base)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ieg_2020 = st.number_input("IEG (Ano Base)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ipv_2020 = st.number_input("IPV (Ano Base)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    iaa_2020 = st.number_input("IAA (Ano Base)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)

with col2:
    ian_2020 = st.number_input("IAN (Ano Base)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ips_2020 = st.number_input("IPS (Ano Base)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ipp_2020 = st.number_input("IPP (Ano Base)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    anos_pm = st.number_input("Anos na Passos Mágicos", min_value=0.0, max_value=15.0, value=2.0, step=0.5)

st.markdown("#### Ano Seguinte (ex: 2021)")
col3, col4 = st.columns(2)

with col3:
    inde_2021 = st.number_input("INDE (Ano Seguinte)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ida_2021 = st.number_input("IDA (Ano Seguinte)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ieg_2021 = st.number_input("IEG (Ano Seguinte)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ipv_2021 = st.number_input("IPV (Ano Seguinte)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)

with col4:
    iaa_2021 = st.number_input("IAA (Ano Seguinte)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ian_2021 = st.number_input("IAN (Ano Seguinte)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ips_2021 = st.number_input("IPS (Ano Seguinte)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ipp_2021 = st.number_input("IPP (Ano Seguinte)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)

# =====================================================
# Botão de predição
# =====================================================
if st.button("Calcular Probabilidade de Risco", type="primary"):

    # Calcular os deltas
    delta_inde = inde_2021 - inde_2020
    delta_ida  = ida_2021  - ida_2020
    delta_ieg  = ieg_2021  - ieg_2020
    delta_ipv  = ipv_2021  - ipv_2020
    delta_iaa  = iaa_2021  - iaa_2020
    delta_ian  = ian_2021  - ian_2020
    delta_ips  = ips_2021  - ips_2020
    delta_ipp  = ipp_2021  - ipp_2020

    # Criar o DataFrame na ordem exata das features do modelo
    dados = pd.DataFrame([[
        inde_2020, ida_2020, ieg_2020, ipv_2020,
        iaa_2020, ian_2020, ips_2020, ipp_2020,
        delta_inde, delta_ida, delta_ieg, delta_ipv,
        delta_iaa, delta_ian, delta_ips, delta_ipp,
        anos_pm
    ]], columns=features)

    # Fazer a predição
    probabilidade = modelo.predict_proba(dados)[0][1]
    classificacao = "Em Risco" if probabilidade >= threshold else "Sem Risco"

    # Exibir resultados
    st.markdown("---")
    st.subheader("Resultado da Predição")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Probabilidade de Risco", f"{probabilidade:.1%}")

    with col_b:
        if classificacao == "Em Risco":
            st.error(f"Classificação: **{classificacao}**")
        else:
            st.success(f"Classificação: **{classificacao}**")

    st.progress(float(probabilidade))
    st.caption(f"Threshold utilizado: {threshold}")