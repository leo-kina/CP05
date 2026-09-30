# CheckPoint05 — Random Forest, XGBoost e LightGBM
### Previsão de faixa de renda (*Adult / Census Income*) — classificação binária

Projeto completo de *machine learning* para a disciplina **Data Science &
Statistical Computing** (FIAP): da definição do problema ao *deploy*, comparando
três algoritmos baseados em árvores sob um protocolo experimental único.

---

## 1. Problema

Estimar se um indivíduo tem **renda anual superior a US\$ 50 mil** (`>50K`) a
partir de atributos sociodemográficos e ocupacionais do Censo dos EUA de 1994.
É uma **classificação binária** com classe positiva `>50K`.

- **Base:** *Adult / Census Income* — UCI / OpenML (id 1590, versão 2),
  48.842 linhas × 15 colunas. Versionada em [`data/adult.csv`](data/adult.csv).
- **Métrica principal:** **ROC-AUC** (base desbalanceada, ~24% positivos).
- **Métricas auxiliares:** PR-AUC, precisão, recall, F1, acurácia.

## 2. Metodologia

1. **Data wrangling:** remoção de duplicatas exatas; diagnóstico de ausentes,
   inconsistências e valores *top-coded*; imputação e *one-hot* feitos **dentro
   do pipeline** (sem *data leakage*).
2. **Desenho experimental:** *holdout* estratificado 80/20 (`random_state=42`),
   `StratifiedKFold`(5). O teste fica **isolado** até a avaliação final.
3. **Baselines** de Random Forest, XGBoost e LightGBM.
4. **Tuning** com **Grid Search** e **Optuna** (20 *trials*/modelo).
5. **Seleção** do melhor modelo por validação cruzada + curva de aprendizado.
6. **Teste final** único + teste de consistência + *deploy*.

## 3. Resultados (conjunto de teste)

<!-- RESULTS_START -->
**ROC-AUC de validação cruzada (5 folds) — 9 configurações:**

| Modelo | Baseline | Grid Search | Optuna |
|---|:--:|:--:|:--:|
| Random Forest | 0,8921 | 0,9165 | 0,9169 |
| XGBoost | 0,9246 | 0,9296 | **0,9299** |
| LightGBM | 0,9268 | 0,9283 | 0,9299 |

**Modelo final selecionado:** **XGBoost (Optuna)** — AUC de validação **0,9299**.

**Desempenho no conjunto de teste (usado uma única vez):**

| Métrica | Valor |
|---|:--:|
| ROC-AUC (principal) | **0,9270** |
| PR-AUC (avg. precision) | 0,8222 |
| Acurácia | 0,8727 |
| Precisão (>50K) | 0,7828 |
| Recall (>50K) | 0,6481 |
| F1 (>50K) | 0,7091 |

Diferença validação → teste de apenas **0,0029** em ROC-AUC → **generalização
confirmada**, sem sobreajuste.
<!-- RESULTS_END -->

## 4. Estrutura do repositório

```
CKP05/
├── Checkpoint05_RF_XGBoost_LightGBM.ipynb   # notebook completo e executado
├── app.py                                   # aplicação Streamlit
├── requirements.txt                         # dependências (versões testadas)
├── README.md
├── data/
│   └── adult.csv                            # base versionada (reprodutibilidade)
└── model/
    ├── final_pipeline.joblib                # pipeline final (pré-proc + modelo)
    └── model_card.json                      # schema, hiperparâmetros e métricas
```

## 5. Como executar

### 5.1 Ambiente

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Notebook

Abra `Checkpoint05_RF_XGBoost_LightGBM.ipynb` e execute **do início ao fim**
(ou `jupyter nbconvert --to notebook --execute --inplace
Checkpoint05_RF_XGBoost_LightGBM.ipynb`). Isso regenera o modelo em `model/`.

### 5.3 Aplicação Streamlit

```bash
streamlit run app.py
```

A interface usa **o mesmo pipeline** salvo pelo notebook, garantindo paridade
entre a previsão do notebook e a da aplicação.

## 6. Reprodutibilidade

- Sementes fixas (`random_state=42`) em *split*, CV, modelos e Optuna.
- Base versionada em `data/`; caso ausente, o notebook a baixa do OpenML.
- Toda transformação que aprende parâmetros está no *pipeline*, ajustada apenas
  no treino.

## 7. Links de entrega

- **GitHub:** https://github.com/leo-kina/CP05
- **Streamlit:** https://cp05-renda.streamlit.app

## 8. Integrantes

| Nome | RM |
|---|---|
| Leonardo Eiji Kina | 562784 |
| Nicholas Braga de Souza | 561733 |
| Tomé Rossi Giani | 562422 |
| Vitor Ramos de Farias | 561958 |


## Referências

Breiman (2001), *Random Forests*; Chen & Guestrin (2016), *XGBoost*;
Ke et al. (2017), *LightGBM*; Akiba et al. (2019), *Optuna*.
Base *Adult* — Becker & Kohavi, UCI Machine Learning Repository.
