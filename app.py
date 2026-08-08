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


# 1. Tipo de aluno

st.subheader("1. Tipo de Aluno")
tipo_aluno = st.radio(
    "O aluno tem pelo menos 2 anos de histórico no programa?",
    ["Não — Aluno novo (sem histórico)", "Sim — Aluno com histórico (2 anos ou mais)"],
    index=0
)


# 2. Indicadores atuais

st.subheader("2. Indicadores do Aluno (Ano Atual)")

col1, col2 = st.columns(2)
with col1:
    ida = st.number_input("IDA", 0.0, 10.0, 7.0, 0.1)
    ieg = st.number_input("IEG", 0.0, 10.0, 7.0, 0.1)
    iaa = st.number_input("IAA", 0.0, 10.0, 7.0, 0.1)
    ips = st.number_input("IPS", 0.0, 10.0, 7.0, 0.1)

with col2:
    ipp = st.number_input("IPP", 0.0, 10.0, 7.0, 0.1)
    ipv = st.number_input("IPV", 0.0, 10.0, 7.0, 0.1)
    fase_num = st.number_input("Fase (número)", 0, 8, 3, 1)

    # Só aparece se o modelo sem delta precisar
    anos_no_programa = None
    if "Anos_No_Programa" in modelo_sem.get("features", []):
        anos_no_programa = st.number_input(
            "Anos no Programa", 0.0, 15.0, 1.0, 0.5
        )


# 3. Deltas (só para aluno com histórico)

if tipo_aluno.startswith("Sim"):
    st.subheader("3. Variação em relação ao ano anterior (Deltas)")
    st.caption("Informe a variação em relação ao ano passado (pode ser negativa).")

    col3, col4 = st.columns(2)
    with col3:
        delta_ida = st.number_input("Delta IDA", -5.0, 5.0, 0.0, 0.1)
        delta_ieg = st.number_input("Delta IEG", -5.0, 5.0, 0.0, 0.1)
        delta_iaa = st.number_input("Delta IAA", -5.0, 5.0, 0.0, 0.1)
    with col4:
        delta_ips = st.number_input("Delta IPS", -5.0, 5.0, 0.0, 0.1)
        delta_ipp = st.number_input("Delta IPP", -5.0, 5.0, 0.0, 0.1)
        delta_ipv = st.number_input("Delta IPV", -5.0, 5.0, 0.0, 0.1)


# Botão de predição

if st.button("Calcular Probabilidade de Risco", type="primary"):

    if tipo_aluno.startswith("Não"):
        # Monta dict e reordena conforme o modelo espera
        valores = {
            "IDA": ida,
            "IEG": ieg,
            "IAA": iaa,
            "IPS": ips,
            "IPP": ipp,
            "IPV": ipv,
            "Fase_Num": fase_num,
        }
        if "Anos_No_Programa" in modelo_sem["features"]:
            valores["Anos_No_Programa"] = anos_no_programa if anos_no_programa is not None else 1.0

        dados = pd.DataFrame([valores])[modelo_sem["features"]]
        probabilidade = modelo_sem["modelo"].predict_proba(dados)[0][1]
        threshold = modelo_sem["threshold"]
        modelo_usado = "Sem Delta (Aluno Novo)"

    else:
        valores = {
            "IDA": ida,
            "IEG": ieg,
            "IAA": iaa,
            "IPS": ips,
            "IPP": ipp,
            "IPV": ipv,
            "Fase_Num": fase_num,
            "delta_IDA": delta_ida,
            "delta_IEG": delta_ieg,
            "delta_IAA": delta_iaa,
            "delta_IPS": delta_ips,
            "delta_IPP": delta_ipp,
            "delta_IPV": delta_ipv,
        }
        dados = pd.DataFrame([valores])[modelo_com["features"]]
        probabilidade = modelo_com["modelo"].predict_proba(dados)[0][1]
        threshold = modelo_com["threshold"]
        modelo_usado = "Com Delta (Aluno com Histórico)"

    classificacao = "Em Risco" if probabilidade >= threshold else "Sem Risco"

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
