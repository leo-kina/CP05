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

# Tradução APENAS para exibição. O valor real (em inglês) continua sendo
# enviado ao modelo, pois o pipeline foi treinado com essas strings; usamos
# format_func para exibir PT sem alterar o dado que o one-hot encoder espera.
trad_valores = {
    "workclass": {
        "Federal-gov": "Governo federal",
        "Local-gov": "Governo municipal",
        "Never-worked": "Nunca trabalhou",
        "Private": "Setor privado",
        "Self-emp-inc": "Autônomo (empresa constituída)",
        "Self-emp-not-inc": "Autônomo (sem empresa)",
        "State-gov": "Governo estadual",
        "Without-pay": "Sem remuneração",
    },
    "education": {
        "10th": "Fundamental/médio (10ª série – EUA)",
        "11th": "Ensino médio (11ª série – EUA)",
        "12th": "Ensino médio (12ª série – EUA)",
        "1st-4th": "1ª a 4ª série",
        "5th-6th": "5ª a 6ª série",
        "7th-8th": "7ª a 8ª série",
        "9th": "9ª série",
        "Assoc-acdm": "Tecnólogo (acadêmico)",
        "Assoc-voc": "Tecnólogo (profissionalizante)",
        "Bachelors": "Bacharelado",
        "Doctorate": "Doutorado",
        "HS-grad": "Ensino médio completo",
        "Masters": "Mestrado",
        "Preschool": "Pré-escola",
        "Prof-school": "Escola profissional (Direito/Medicina)",
        "Some-college": "Superior incompleto",
    },
    "marital-status": {
        "Divorced": "Divorciado(a)",
        "Married-AF-spouse": "Casado(a) — cônjuge militar",
        "Married-civ-spouse": "Casado(a) — cônjuge civil",
        "Married-spouse-absent": "Casado(a) — cônjuge ausente",
        "Never-married": "Solteiro(a)",
        "Separated": "Separado(a)",
        "Widowed": "Viúvo(a)",
    },
    "occupation": {
        "Adm-clerical": "Administrativo/burocrático",
        "Armed-Forces": "Forças Armadas",
        "Craft-repair": "Ofícios e reparos",
        "Exec-managerial": "Executivo/gerencial",
        "Farming-fishing": "Agricultura e pesca",
        "Handlers-cleaners": "Serviços gerais/limpeza",
        "Machine-op-inspct": "Operador/inspetor de máquinas",
        "Other-service": "Outros serviços",
        "Priv-house-serv": "Serviço doméstico",
        "Prof-specialty": "Profissional especializado",
        "Protective-serv": "Segurança/proteção",
        "Sales": "Vendas",
        "Tech-support": "Suporte técnico",
        "Transport-moving": "Transporte/movimentação",
    },
    "relationship": {
        "Husband": "Marido",
        "Not-in-family": "Fora da família",
        "Other-relative": "Outro parente",
        "Own-child": "Filho(a)",
        "Unmarried": "Não casado(a)",
        "Wife": "Esposa",
    },
    "race": {
        "Amer-Indian-Eskimo": "Indígena americano/Esquimó",
        "Asian-Pac-Islander": "Asiático/Ilhas do Pacífico",
        "Black": "Negro(a)",
        "Other": "Outra",
        "White": "Branco(a)",
    },
    "sex": {
        "Female": "Feminino",
        "Male": "Masculino",
    },
    "native-country": {
        "Cambodia": "Camboja",
        "Canada": "Canadá",
        "China": "China",
        "Columbia": "Colômbia",
        "Cuba": "Cuba",
        "Dominican-Republic": "República Dominicana",
        "Ecuador": "Equador",
        "El-Salvador": "El Salvador",
        "England": "Inglaterra",
        "France": "França",
        "Germany": "Alemanha",
        "Greece": "Grécia",
        "Guatemala": "Guatemala",
        "Haiti": "Haiti",
        "Holand-Netherlands": "Holanda (Países Baixos)",
        "Honduras": "Honduras",
        "Hong": "Hong Kong",
        "Hungary": "Hungria",
        "India": "Índia",
        "Iran": "Irã",
        "Ireland": "Irlanda",
        "Italy": "Itália",
        "Jamaica": "Jamaica",
        "Japan": "Japão",
        "Laos": "Laos",
        "Mexico": "México",
        "Nicaragua": "Nicarágua",
        "Outlying-US(Guam-USVI-etc)": "Territórios dos EUA (Guam, USVI etc.)",
        "Peru": "Peru",
        "Philippines": "Filipinas",
        "Poland": "Polônia",
        "Portugal": "Portugal",
        "Puerto-Rico": "Porto Rico",
        "Scotland": "Escócia",
        "South": "Coreia do Sul",
        "Taiwan": "Taiwan",
        "Thailand": "Tailândia",
        "Trinadad&Tobago": "Trinidad e Tobago",
        "United-States": "Estados Unidos",
        "Vietnam": "Vietnã",
        "Yugoslavia": "Iugoslávia",
    },
}
cat_items = list(cat_schema.items())
for i, (c, opcoes) in enumerate(cat_items):
    alvo_col = col1 if i % 2 == 0 else col2
    with alvo_col:
        entradas[c] = st.selectbox(
            rotulos_cat.get(c, c),
            opcoes,
            format_func=lambda v, _c=c: trad_valores.get(_c, {}).get(v, v),
        )

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
