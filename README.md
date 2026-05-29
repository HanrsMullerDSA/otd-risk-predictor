# 📦 OTD Risk Predictor
### Predição de Atraso Logístico no Momento da Compra

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![Model](https://img.shields.io/badge/Model-v2.0-purple)

> Aplicação interativa que estima, **no momento da compra**, a probabilidade de um pedido atrasar — antes de qualquer evento logístico ocorrer — permitindo intervenção preventiva na operação.

🔗 **[Acessar o app ao vivo](https://otd-risk-predictor-hanrssilveira.streamlit.app)**

---

## 🎯 Objetivo do Projeto

OTD (On Time Delivery) mede se um pedido foi entregue no prazo prometido ao cliente. Este projeto constrói um modelo preditivo capaz de sinalizar, **no momento da compra**, quais pedidos têm maior risco de atraso para permitir ações preventivas e apoiar a melhoria do OTIF.

**Pergunta central:** dado o que sabemos no momento da compra — origem, destino, produto, prazo e histórico da rota — qual a probabilidade deste pedido atrasar?

---

## 🗂️ Estrutura do Projeto

```
otd-risk-predictor/
├── app.py                              # Aplicação Streamlit
├── requirements.txt                    # Dependências
├── notebooks/
│   └── otd_predictor_supply_chain_analytics.ipynb  # Notebook end-to-end
└── artifacts/
    ├── modelo_balanceado.joblib        # Modelo XGBoost (pipeline completo)
    ├── lookup_route.parquet            # Histórico causal por rota
    ├── lookup_seller.parquet           # Histórico causal por seller
    ├── geo_median.parquet              # Lat/lon mediana por prefixo de CEP
    ├── encoders.json                   # Mapeamento categoria → inteiro
    └── metadata.json                   # Features, threshold, métricas e versão
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
| Peso, preço e valor do frete | `seller_lead_time` final |
| Sazonalidade (mês, dia, weekend) | Qualquer resultado do pedido |
| Distância geográfica (haversine) | — |

### Engenharia Temporal Causal
As features históricas (`hist_atraso_route_t`, `hist_atraso_seller_t`) foram calculadas com **merge_asof** — para cada pedido, o histórico considera apenas pedidos cujo resultado (`label_known_at`) era observável antes do momento da compra (`order_purchase_timestamp`). Nenhum dado futuro contamina o cálculo.

O **Bayesian Smoothing** (α=30 para rotas, α=10 para sellers) ancora entidades com poucos pedidos ao prior global, evitando estimativas instáveis (0% ou 100%) em rotas ou sellers raros.

### Pipeline de Modelagem

| Etapa | Decisão técnica |
|---|---|
| Algoritmo | XGBoost Classifier |
| Balanceamento | `scale_pos_weight` proporcional ao desbalanceamento da classe |
| Split | Temporal 80/20 — treino com pedidos mais antigos, teste com pedidos futuros |
| Tuning | `RandomizedSearchCV` com `TimeSeriesSplit` (3 folds temporais), otimizando PR-AUC |
| Calibração | `CalibratedClassifierCV` com regressão isotônica (`cv="prefit"`) |
| Threshold | Calibrado para maximizar F1-score da classe de atraso |

### Por que PR-AUC e não ROC-AUC como métrica de seleção?
Com apenas **3,49% de atrasos no conjunto de teste**, o baseline aleatório equivale a PR-AUC ≈ 0,035. A ROC-AUC pode inflar resultados em datasets desbalanceados e mascarar baixo recall real — a PR-AUC penaliza diretamente baixa precision e é sensível ao desbalanceamento, revelando o ganho real entre os modelos.

### Performance dos Modelos

| Modelo | ROC-AUC | PR-AUC |
|--------|---------|--------|
| Inicial (baseline) | 0.6595 | 0.0596 |
| Balanceado | 0.6153 | 0.0552 |
| **Otimizado + calibrado (selecionado)** | **0.6348** | **0.0575** |
| Recency weighted + calibrado | 0.6188 | 0.0524 |

> O modelo entrega ganho real frente ao baseline aleatório (PR-AUC ≈ 0,035), mas com limitação estrutural: atrasos são causados por eventos operacionais pontuais (falha de transportadora, pico regional, capacidade de armazém) que não existem no momento da compra. Features estruturais capturam padrões lentos — não eventos imprevistos.

### Distribuição Temporal do Treino/Teste

| Conjunto | Taxa de Atraso | OTIF |
|---|---|---|
| Treino (80%) | 7,59% | 92,41% |
| Teste (20%) | 3,49% | 96,51% |

A diferença de ~4 p.p. reflete melhoria real da operação ao longo do tempo — esperada em dados temporais de supply chain.

### Features do Modelo (19 variáveis)

| Feature | Tipo | Descrição |
|---|---|---|
| `month` | Temporal | Mês da compra — sazonalidade e picos de demanda |
| `weekday` | Temporal | Dia da semana da compra |
| `weekend` | Temporal | Flag: compra realizada no fim de semana |
| `time_to_buy` | Temporal | Engenharia temporal do momento da compra |
| `hist_atraso_route_smooth` | Histórica | Taxa de atraso da rota com Bayesian smoothing (α=30) |
| `hist_atraso_route_t` | Histórica | Taxa bruta de atraso da rota (anti-leakage) |
| `hist_atraso_seller_smooth` | Histórica | Taxa de atraso do seller com smoothing (α=10) |
| `hist_atraso_seller_t` | Histórica | Taxa bruta de atraso do seller (anti-leakage) |
| `customer_state` | Geográfica | Estado do cliente (destino) |
| `seller_state` | Geográfica | Estado do vendedor (origem) |
| `distancia_km` | Logística | Distância haversine entre seller e customer |
| `prazo_prometido_dias` | Operacional | Dias entre compra e prazo estimado de entrega |
| `freight_value` | Financeira | Valor do frete cobrado |
| `price` | Financeira | Preço do produto |
| `product_weight_g` | Produto | Peso do produto em gramas |
| `product_category_name` | Produto | Categoria do produto |
| `seller_id` | Operacional | Identificador do seller (codificado) |
| `route` | Geográfica | Rota origem-destino (`SELLER_STATE_CUSTOMER_STATE`) |
| `volume_seller_7d_t` | Operacional | Volume de pedidos do seller nos últimos 7 dias (anti-leakage) |

### Threshold de Decisão
- **Threshold ótimo:** `0.3715` — calibrado para maximizar F1-score da classe de atraso
- **Prior global de atraso:** `7,24%` — usado como fallback (cold start) para rotas/sellers sem histórico

---

## 📱 Funcionalidades do App

### 🔍 Aba — Previsão de Risco
- Selecione estado de origem (vendedor) e destino (cliente)
- Informe peso do produto, valor do frete e prazo prometido
- O modelo retorna a probabilidade de atraso com gauge visual
- Classificação em 4 níveis: **Baixo · Moderado · Alto · Crítico**
- Ações operacionais sugeridas quando acima do threshold
- Painel de parâmetros e sinais históricos da rota

### 🗺️ Aba — Mapa de Risco por Estado
- Treemap interativo com taxa histórica de atraso por estado de destino
- KPIs automáticos: estado mais crítico, mais seguro, taxa média
- Top 15 rotas com maior risco histórico (com taxa suavizada)
- Insights automáticos gerados a partir dos dados reais
- Expander com metodologia detalhada

### 👤 Aba — Sobre o Projeto
- Contexto completo do projeto e problema de negócio
- Stack técnica e destaques metodológicos
- Performance do modelo e link para o repositório

---

## 🖥️ Como Executar Localmente

### Pré-requisitos
- Python 3.11+
- Os arquivos da pasta `artifacts/`

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

## 🔄 Como Reproduzir os Artefatos

Os artefatos da pasta `artifacts/` foram gerados a partir do dataset público da Olist.

**1. Baixe o dataset**
Acesse [Olist E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) no Kaggle e baixe os CSVs.

**2. Configure o caminho no notebook**
No notebook `notebooks/otd_predictor_supply_chain_analytics.ipynb`, atualize a variável de caminho para os CSVs na seção de setup (célula 3).

**3. Execute o notebook completo**
Execute todas as células em ordem. A seção **15. Exportação de Artefatos** gera automaticamente todos os arquivos necessários:

| Artefato | Conteúdo |
|---|---|
| `modelo_balanceado.joblib` | XGBoost otimizado + calibrador isotônico (pipeline) |
| `encoders.json` | Mapeamento categoria → inteiro por coluna |
| `lookup_route.parquet` | Histórico causal de atraso por rota |
| `lookup_seller.parquet` | Histórico causal de atraso e volume por seller |
| `geo_median.parquet` | Lat/lon mediana por prefixo de CEP |
| `metadata.json` | Features, threshold ótimo, prior global e métricas |

---

## 🚀 Deploy

O app está publicado no **Streamlit Cloud** com deploy contínuo — qualquer push para a branch `main` atualiza o app automaticamente.

🔗 [otd-risk-predictor-hanrssilveira.streamlit.app](https://otd-risk-predictor-hanrssilveira.streamlit.app)

---

## 👤 Autor

**Hanrs Muller Lima da Silveira**

Profissional de Supply Chain com foco em Analytics e Data Science aplicados a operações logísticas. Este projeto faz parte do **Supply Chain Analytics Lab** — iniciativa de desenvolvimento e portfólio em ciência de dados para cadeia de suprimentos.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-hanrsmuller-blue?logo=linkedin)](https://www.linkedin.com/in/hanrsmuller/)
[![Gmail](https://img.shields.io/badge/Gmail-hanrs.silveira@gmail.com-red?logo=gmail)](mailto:hanrs.silveira@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-HanrsMullerDSA-black?logo=github)](https://github.com/HanrsMullerDSA)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.