# -*- coding: utf-8 -*-
"""
CheckPoint05 — Aplicação Streamlit
Previsão de faixa de renda (>50K) com o mesmo pipeline treinado no notebook.

Executar localmente:
    streamlit run app.py

A aplicação carrega o pipeline final (model/final_pipeline.joblib) — que já
contém TODA a preparação de dados (imputação + one-hot) — garantindo paridade
total entre o notebook e a interface.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# joblib.load carrega um artefato PRODUZIDO POR ESTE PROJETO (confiável),
# gerado pelo próprio notebook — não é conteúdo de fonte externa.
import joblib

MODEL_DIR = Path(__file__).parent / "model"
PIPE_PATH = MODEL_DIR / "final_pipeline.joblib"
CARD_PATH = MODEL_DIR / "model_card.json"

st.set_page_config(page_title="Previsão de Renda — CheckPoint05",
                   page_icon="💰", layout="centered")


@st.cache_resource
def carregar_modelo():
    pipe = joblib.load(PIPE_PATH)
    with open(CARD_PATH, encoding="utf-8") as f:
        card = json.load(f)
    return pipe, card


st.title("💰 Previsão de faixa de renda (>50K)")
st.caption("Random Forest · XGBoost · LightGBM — CheckPoint05 (Data Science)")

if not PIPE_PATH.exists() or not CARD_PATH.exists():
    st.error(
        "Artefatos do modelo não encontrados. Execute o notebook "
        "`Checkpoint05_RF_XGBoost_LightGBM.ipynb` até o final (Exercício 7) "
        "para gerar `model/final_pipeline.joblib` e `model/model_card.json`."
    )
    st.stop()

pipe, card = carregar_modelo()

with st.expander("ℹ️ Sobre o modelo", expanded=False):
    st.write(f"**Modelo final:** {card.get('modelo_final', '—')}")
    st.write(f"**Alvo:** {card.get('target', '—')}")
    st.write("**Desempenho no conjunto de teste:**")
    st.json(card.get("metricas_teste", {}))
    st.write("**Hiperparâmetros:**")
    st.json(card.get("best_params", {}))

st.subheader("Informe os dados do indivíduo")

col1, col2 = st.columns(2)
entradas = {}

# --- Campos numéricos ---
num_schema = card["numeric"]
rotulos_num = {
    "age": "Idade (anos)",
    "capital-gain": "Ganhos de capital (US$)",
    "capital-loss": "Perdas de capital (US$)",
    "hours-per-week": "Horas trabalhadas por semana",
}
for i, (c, info) in enumerate(num_schema.items()):
    alvo_col = col1 if i % 2 == 0 else col2
    with alvo_col:
        entradas[c] = st.number_input(
            rotulos_num.get(c, c),
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["median"]),
            step=1.0,
        )

# --- Campos categóricos ---
cat_schema = card["categorical"]
rotulos_cat = {
    "workclass": "Tipo de empregador",
    "education": "Escolaridade",
    "marital-status": "Estado civil",
    "occupation": "Ocupação",
    "relationship": "Papel na família",
    "race": "Raça",
    "sex": "Sexo",
    "native-country": "País de origem",
}
cat_items = list(cat_schema.items())
for i, (c, opcoes) in enumerate(cat_items):
    alvo_col = col1 if i % 2 == 0 else col2
    with alvo_col:
        entradas[c] = st.selectbox(rotulos_cat.get(c, c), opcoes)

st.divider()

if st.button("🔮 Prever faixa de renda", type="primary", use_container_width=True):
    # monta a linha exatamente com as colunas usadas no treino
    colunas = card["num_cols"] + card["cat_cols"]
    X_novo = pd.DataFrame([{c: entradas[c] for c in colunas}])

    proba = float(pipe.predict_proba(X_novo)[0, 1])
    threshold = card.get("threshold", 0.5)
    classe = ">50K" if proba >= threshold else "<=50K"

    st.subheader("Resultado")
    if classe == ">50K":
        st.success(f"### Previsão: renda **> US\\$ 50 mil/ano**")
    else:
        st.info(f"### Previsão: renda **≤ US\\$ 50 mil/ano**")

    st.metric("Probabilidade estimada de renda > 50K", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(f"Limiar de decisão: {threshold:.2f} · Classe positiva: >50K")

    with st.expander("Ver dados enviados ao modelo"):
        st.dataframe(X_novo.T.rename(columns={0: "valor"}))
