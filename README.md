# Previsão de Churn de Clientes de E-commerce

> Atividade Avaliativa Final — Disciplina de Inteligência Computacional  
> Faculdade de Tecnologia de Jundiaí · Curso Superior de Tecnologia em Ciência de Dados

---

## Sobre o projeto

Este projeto desenvolve um **fluxo completo de Machine Learning** para prever o *churn* (cancelamento) de clientes de uma plataforma de e-commerce. O objetivo é identificar, com antecedência, quais clientes têm maior probabilidade de abandonar a plataforma possibilitando ações preventivas de retenção.

O trabalho cobre todas as etapas de um pipeline de Ciência de Dados: análise exploratória, pré-processamento, engenharia de atributos, modelagem, validação cruzada, otimização de hiperparâmetros e avaliação final.

---

## Dataset

| Item | Detalhe |
|---|---|
| **Fonte** | [Ecommerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) — Kaggle |
| **Registros** | 5.630 clientes |
| **Variável-alvo** | `Churn` (1 = cancelou, 0 = permaneceu) |
| **Desafios** | 7 colunas com valores ausentes, categorias inconsistentes, classes desbalanceadas (~83% / ~17%) |

**Principais variáveis:** `Tenure`, `Complain`, `SatisfactionScore`, `PreferredLoginDevice`, `PreferredPaymentMode`, `CashbackAmount`, `DaySinceLastOrder`, entre outras.

---

## Solução desenvolvida

### Tecnologias

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Pipeline-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)

### Estrutura do fluxo

```
Dataset (Kaggle)
    │
    ▼
Análise Exploratória (EDA)
    │  histogramas · boxplots · heatmap · scatterplot
    │
    ▼
Pré-processamento
    │  padronização de categorias inconsistentes
    │
    ▼
Engenharia de Atributos (4 novos atributos)
    │
    ▼
Divisão Treino / Teste  (80% / 20%, stratify=y)
    │
    ▼
Pipeline + ColumnTransformer
    │  trilha numérica: SimpleImputer (mediana) → StandardScaler
    │  trilha categórica: SimpleImputer (moda) → OneHotEncoder
    │
    ▼
StratifiedKFold (k=5) + GridSearchCV
    │
    ▼
Avaliação Final (acurácia + matriz de confusão)
```

### Pré-processamento

| Tipo de variável | Ausentes | Escala | Codificação |
|---|---|---|---|
| Numéricas | `SimpleImputer(strategy='median')` | `StandardScaler` | — |
| Categóricas | `SimpleImputer(strategy='most_frequent')` | — | `OneHotEncoder(handle_unknown='ignore')` |

Todo o pré-processamento foi encapsulado no `Pipeline` para **prevenir *data leakage***: os parâmetros de imputação e escalonamento são aprendidos **apenas nos dados de treino** de cada *fold*.

### Engenharia de Atributos

Foram criados **4 novos atributos** com justificativa de negócio:

| Atributo | Cálculo | Motivação |
|---|---|---|
| `CouponPerOrder` | `CouponUsed / (OrderCount + 1)` | Mede a dependência de promoções |
| `CashbackPerOrder` | `CashbackAmount / (OrderCount + 1)` | Benefício médio percebido por compra |
| `InactivityRatio` | `DaySinceLastOrder / (Tenure + 1)` | Cliente antigo que parou de comprar = sinal de churn |
| `TenureGroup` | faixas de `Tenure` | Segmenta clientes em Novo / Intermediário / Consolidado / Veterano |

---

## Resultados

### Validação cruzada (StratifiedKFold, k=5)

| Fold | Acurácia |
|---|---|
| Fold 1 | 81,13% |
| Fold 2 | 80,47% |
| Fold 3 | 81,91% |
| Fold 4 | 81,69% |
| Fold 5 | 81,44% |
| **Média** | **81,33%** |
| Desvio padrão | ±0,50 pp |

### GridSearchCV — melhor configuração

| Parâmetro | Valor escolhido |
|---|---|
| `solver` | `liblinear` |
| `penalty` | `l1` |
| `C` | `1` |
| Acurácia CV | **81,42%** |

### Avaliação final no conjunto de teste

| Métrica | Valor |
|---|---|
| **Acurácia** | **80,82%** |
| Recall — Cancelou | **84%** |
| Precision — Cancelou | 46% |
| F1-score — Cancelou | 0,60 |
| Recall — Permaneceu | 80% |
| Precision — Permaneceu | 96% |

**Matriz de confusão (1.126 clientes no teste):**

|  | Previsto: Permaneceu | Previsto: Cancelou |
|---|---|---|
| **Real: Permaneceu** (936) | 749  | 187 |
| **Real: Cancelou** (190) | 30 | 160  |

O modelo identificou **160 dos 190 churners reais (recall de 84%)**, com apenas 30 clientes em risco não detectados. A configuração `class_weight='balanced'` foi decisiva para que o modelo não ignorasse a classe minoritária.

---

##  Principais conclusões

- A diferença entre a acurácia de validação (81,33%) e a de teste (80,82%) é inferior a 1 ponto percentual, confirmando **boa generalização e ausência de overfitting**.
- O atributo `InactivityRatio` mostrou a maior separação visual entre churners e não-churners, evidenciando que combinar *recência* com *maturidade* do cliente gera um sinal de "esfriamento" que nenhuma coluna isolada expressa.
- Clientes `Novos` (até 6 meses) apresentaram taxa de churn de **32,4%** — mais de 5× a taxa dos `Intermediários` (5,7%) — e nenhum `Veterano` cancelou no período analisado.
- O pipeline organizou o fluxo de forma reprodutível, permitindo que qualquer nova base de clientes seja processada e pontuada pelo mesmo objeto sem reexecutar etapas manuais.

---

##  Como executar

**Pré-requisitos:** Python 3.10+, Jupyter Notebook

```bash
# 1. Clone o repositório
git clone https://github.com/<DaianneS>/<Int_Computacional_previsao_churn
>.git
cd <nome-do-repo>

# 2. Instale as dependências
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn openpyxl

# 3. Abra o notebook
jupyter notebook INTCOM_1_final.ipynb
```

> O dataset é baixado automaticamente via `kagglehub` na primeira execução (requer conta no Kaggle configurada localmente).

---

## Estrutura do repositório

```
.
├── INTCOM_1_final.ipynb   # Notebook principal com todo o fluxo
└── README.md              # Este arquivo
```

---

## Autoras

| | |
|---|---|
| **Daianne Soares Silva** | [![GitHub](https://img.shields.io/badge/GitHub-perfil-black?logo=github)]([https://github.com](https://github.com/DaianneS)/) |
| **Laiana Naiara Souza Reis** | [![GitHub](https://img.shields.io/badge/GitHub-perfil-black?logo=github)]([https://github.com/](https://github.com/laianareis)) |

---

## Informações acadêmicas

- **Instituição:** Faculdade de Tecnologia de Jundiaí (FATEC Jundiaí)
- **Curso:** Tecnólogo em Ciência de Dados
- **Disciplina:** Inteligência Computacional
- **Professor:** Me. Mateus Guilherme Fuini
- **Período:** 1º semestre de 2026
