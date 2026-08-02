import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# Configuração da página
# =====================================================
st.set_page_config(
    page_title="Passos Mágicos - Risco de Defasagem",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Predição de Risco de Defasagem")
st.markdown("### Passos Mágicos")
st.write("Insira os indicadores do aluno para estimar a probabilidade de risco.")

# =====================================================
# Carregar o modelo
# =====================================================
@st.cache_resource
def carregar_modelo():
    return joblib.load('modelo_risco_passos.pkl')

modelo = carregar_modelo()

# =====================================================
# Inputs do usuário
# =====================================================
st.subheader("Indicadores do Aluno (Ano Base)")

col1, col2 = st.columns(2)

with col1:
    inde = st.number_input("INDE", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ida = st.number_input("IDA (Aprendizagem)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ieg = st.number_input("IEG (Engajamento)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ipv = st.number_input("IPV (Ponto de Virada)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    iaa = st.number_input("IAA (Autoavaliação)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)

with col2:
    ian = st.number_input("IAN (Adequação ao Nível)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ips = st.number_input("IPS (Psicossocial)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    ipp = st.number_input("IPP (Psicopedagógico)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    anos_pm = st.number_input("Anos na Passos Mágicos", min_value=0.0, max_value=15.0, value=2.0, step=0.5)

# =====================================================
# Botão de predição
# =====================================================
if st.button("Calcular Probabilidade de Risco", type="primary"):
    
    # Criar dataframe com os inputs
    dados = pd.DataFrame([[
        inde, ida, ieg, ipv, iaa, ian, ips, ipp, anos_pm
    ]], columns=[
        'INDE_2020', 'IDA_2020', 'IEG_2020', 'IPV_2020',
        'IAA_2020', 'IAN_2020', 'IPS_2020', 'IPP_2020', 'ANOS_PM_2020'
    ])
    
    # Fazer a predição
    probabilidade = modelo.predict_proba(dados)[0][1]
    classificacao = "Em Risco" if probabilidade >= 0.40 else "Sem Risco"
    
    # Exibir resultados
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
    
    # Barra de probabilidade
    st.progress(float(probabilidade))
    
    st.caption(f"Threshold utilizado: 0.40")