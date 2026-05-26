# 📦 OTD Risk Predictor
### Predição de Atraso Logístico no Momento da Compra

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> Aplicação interativa que estima, **no momento da compra**, a probabilidade de um pedido atrasar — antes de qualquer evento logístico ocorrer — permitindo intervenção preventiva na operação.

🔗 **[Acessar o app ao vivo](https://otd-risk-predictor-hanrssilveira.streamlit.app)**

---

## 🖼️ Screenshots

### Previsão de Risco
![Previsão de Risco](docs/screenshot_previsao.png)

### Mapa de Risco por Estado
![Mapa de Risco](docs/screenshot_mapa.png)

---

## 🎯 Objetivo do Projeto

OTD (On Time Delivery) mede se um pedido foi entregue no prazo prometido ao cliente. Este projeto constrói um modelo preditivo capaz de sinalizar, **no momento da compra**, quais pedidos têm maior risco de atraso para permitir ações preventivas e apoiar a melhoria do OTIF.

**Pergunta central:** dado o que sabemos no momento da compra — origem, destino, produto, prazo e histórico da rota — qual a probabilidade deste pedido atrasar?

---

## 🗂️ Estrutura do Projeto

```
otd-risk-predictor/
├── app.py                        # Aplicação Streamlit
├── requirements.txt              # Dependências
├── docs/
│   ├── screenshot_previsao.png
│   └── screenshot_mapa.png
└── artifacts/
    ├── modelo_balanceado.joblib  # Modelo XGBoost treinado
    ├── lookup_route.parquet      # Histórico causal por rota
    ├── lookup_seller.parquet     # Histórico causal por seller
    └── encoders.json             # Mapeamento de variáveis categóricas
```

---

## 🔬 Metodologia

### Dataset
- **Fonte:** [Olist E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)
- **Volume:** 96.478 pedidos entregues — 2016 a 2018
- **Target:** `target_atraso` — pedido entregue após a data estimada (1 = atrasou, 0 = no prazo)
- **Desbalanceamento:** 93,2% no prazo × 6,8% atrasados

### Princípio Anti-Leakage
O modelo usa **apenas informações disponíveis no momento da compra**. Variáveis pós-evento foram estritamente excluídas:

| ✅ Permitido (momento da compra) | ❌ Excluído (pós-evento) |
|---|---|
| Estado de origem e destino | `transport_lead_time` |
| Histórico causal até t-1 | `time_variance` |
| Prazo prometido ao cliente | Data real de entrega |
| Peso e valor do frete | `seller_lead_time` final |
| Sazonalidade (mês, dia) | Qualquer resultado do pedido |

### Engenharia Temporal Causal
As features históricas (`hist_atraso_route_t`, `hist_atraso_seller_t`) foram calculadas com **merge_asof** — para cada pedido, o histórico considera apenas pedidos cujo resultado (`label_known_at`) era observável antes do momento da compra (`order_purchase_timestamp`). Nenhum dado futuro contamina o cálculo.

O **Bayesian Smoothing** (α=30) ancora rotas com poucos pedidos ao prior global, evitando estimativas instáveis (0% ou 100%) em rotas raras.

### Modelo
- **Algoritmo:** XGBoost Classifier
- **Balanceamento:** `scale_pos_weight ≈ 12` (penaliza erros na classe minoritária)
- **Split:** temporal 80/20 — treino com pedidos mais antigos, teste com pedidos futuros
- **Threshold:** 37,2% — calibrado para maximizar F1-score da classe de atraso

### Performance

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,674 |
| Recall (atrasos) | 38,3% |
| Threshold ótimo | 37,2% |
| Taxa base de atraso | 6,8% |

### Features do Modelo (14 variáveis)

| Feature | Tipo | Importância |
|---|---|---|
| `month` | Temporal | 18,4% |
| `hist_atraso_route_smooth` | Histórica | 11,9% |
| `customer_state` | Geográfica | 10,1% |
| `prazo_prometido_dias` | Operacional | 7,6% |
| `hist_atraso_route_t` | Histórica | 6,7% |
| `distancia_km` | Logística | — |
| `freight_value` | Produto | — |
| `hist_atraso_seller_t` | Histórica | — |
| + 6 outras | — | — |

---

## 🖥️ Como Executar Localmente

### Pré-requisitos
- Python 3.11+
- Os arquivos da pasta `artifacts/` (modelo e tabelas de lookup)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/HanrsMullerDSA/otd-risk-predictor.git
cd otd-risk-predictor

# Instale as dependências
pip install -r requirements.txt

# Execute o app
streamlit run app.py
```

O app abrirá automaticamente em `http://localhost:8501`.

---

## 📱 Funcionalidades do App

### 🔍 Aba — Previsão de Risco
- Selecione estado de origem (vendedor) e destino (cliente)
- Informe peso do produto, valor do frete e prazo prometido
- O modelo retorna a probabilidade de atraso com gauge visual
- Classificação em 4 níveis: Baixo · Moderado · Alto · Crítico
- Ações operacionais sugeridas quando acima do threshold

### 🗺️ Aba — Mapa de Risco por Estado
- Treemap interativo com taxa histórica de atraso por estado de destino
- KPIs automáticos: estado mais crítico, mais seguro, taxa média
- Top 15 rotas com maior risco histórico
- Insights automáticos gerados a partir dos dados reais

---

## 🚀 Deploy

O app está publicado no **Streamlit Cloud** com deploy contínuo — qualquer push para a branch `main` atualiza o app automaticamente.

🔗 [otd-risk-predictor-hanrssilveira.streamlit.app](https://otd-risk-predictor-hanrssilveira.streamlit.app)

---

## 👤 Autor

**Hanrs Muller Lima da Silveira**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-hanrsmuller-blue?logo=linkedin)](https://www.linkedin.com/in/hanrsmuller/)
[![Gmail](https://img.shields.io/badge/Gmail-hanrs.silveira@gmail.com-red?logo=gmail)](mailto:hanrs.silveira@gmail.com)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
