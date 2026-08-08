# 🎯 Previsão de Risco de Defasagem — Passos Mágicos

Modelo de classificação para estimar a probabilidade de um aluno apresentar defasagem, apoiando ações preventivas e prioritárias de acompanhamento pedagógico.

---

## 🎯 Objetivo

Desenvolver um modelo preditivo capaz de identificar alunos em risco de defasagem e realizar uma análise do cenário atual da base da Passos Mágicos, gerando insumos para decisões pedagógicas orientadas a dados.

---

## 📂 Estrutura do Projeto

1. Análise da base  
2. Limpeza e tratamento  
3. Perguntas de negócio  
4. Feature engineering  
5. Modelagem (comparação de algoritmos + treino final)  
6. Avaliação (métricas, threshold e interpretabilidade)  
7. Deploy (Streamlit)

---

## 🤖 Machine Learning

### Análise e preparação dos dados
- Padronização dos indicadores educacionais  
- Tratamento da variável de fase  
- Construção do target de risco: `Defasagem < 0`

### Feature Engineering
- Uso dos indicadores: **IDA, IEG, IAA, IPS, IPP, IPV** e **fase**  
- Exclusão do **IAN** por sobreposição conceitual com a defasagem  
- Criação de **deltas** (evolução ano a ano) para alunos com histórico  

### Estratégia de modelagem dual
| Perfil do aluno | Modelo |
|-----------------|--------|
| Novo (sem histórico de 2 anos) | **Sem delta** |
| Com histórico (2+ anos) | **Com delta** (maior assertividade) |

### Escolha do modelo
Foram comparados:
- Logistic Regression  
- Random Forest  
- XGBoost  
- MLP (rede neural)

O **Random Forest** apresentou o melhor equilíbrio entre desempenho e robustez.

### Threshold
- Valor definido: **0,40**  
- Critério: priorizar **recall** (identificar a maior parte dos alunos em risco)  
- Trade-off aceito: maior número de falsos positivos, alinhado à intervenção precoce  

### Deploy
Aplicação em **Streamlit**, com seleção do tipo de aluno e uso automático do modelo adequado (com ou sem delta).

---

## 🔗 Links

- 📓 [Notebook Python](https://github.com/Ygorca125/Projeto-Previs-o-defasagem-alunos/blob/main/Projeto_M%C3%B3dulo_5_.ipynb)  
- 🎥 Apresentação no YouTube: *(inserir link)*  
- 🚀 [Aplicação Streamlit](https://projeto-previs-o-defasagem-alunos-aowcugb3xv48z72wrkh9qm.streamlit.app/)

---

## 📌 Principais entregas

- Análise exploratória e respostas às perguntas de negócio  
- Modelo preditivo de risco de defasagem  
- Comparação de algoritmos com validação cruzada  
- App interativo para apoio à decisão pedagógica  
