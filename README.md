# 🚨 Sistema de Detecção de Fraudes em Transferências Bancárias

Este projeto utiliza técnicas de aprendizado de máquina para prever possíveis fraudes em transferências financeiras, com base em dados como valor transferido e IP de origem. A solução visa ajudar instituições financeiras e fintechs a identificarem comportamentos anômalos em transações, aumentando a segurança e prevenindo perdas.

---

## 📌 Objetivo

Detectar transações suspeitas de forma automática, utilizando um modelo treinado com dados históricos, e gerar relatórios analíticos e gráficos de apoio à decisão.

---

## 🧠 Bibliotecas Utilizadas

- `pandas` – Manipulação de dados tabulares.
- `scikit-learn` – Treinamento e avaliação de modelos de machine learning (ex: regressão logística, random forest).
- `imbalanced-learn` – Balanceamento de classes com a técnica SMOTE.
- `matplotlib` – Geração de gráficos (ex: curvas ROC, gráficos de barras).
- `seaborn` – Visualização estatística aprimorada.
- `pickle` – Serialização de modelos (nativa do Python).
- `smtplib` / `email.message` – Envio automatizado de relatórios por e-mail (também nativas do Python).

---

## 📦 Instalação das Dependências

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn
'''

##📁 Estrutura do Projeto
.
-├── dados_analise/
-│   ├── novas_contas.csv
-│   ├── novos_usuarios.csv
-│   ├── novas_transferencias.csv
-│   └── log_transferencias.csv       # Gerado após a previsão
-├── modelo_fraude.pkl                # Modelo treinado
-├── scaler_fraude.pkl                # Normalizador usado
-├── treinar_modelo.py               # Script de treinamento
-├── prever_fraude.py                # Script de análise + envio por e-mail
-└── interface_.py      

🔎 Funcionalidades
Pré-processamento dos dados de entrada.
Previsão automatizada de fraudes com limiar ajustável.
Geração de relatórios CSV com resultados categorizados (fraude ou não).
Visualização gráfica:
  Gráfico de barras com total de fraudes e não fraudes.
  Boxplot com distribuição dos valores de transferência por categoria.
  Envio de relatórios por e-mail com autenticação segura.
Interface gráfica com Tkinter para execução facilitada e envio ao vivo durante apresentações.

📊 Geração de Gráficos e Relatórios
Após a análise, o sistema:
Classifica cada operação como Fraude ou Não Fraude.
Gera um gráfico de barras e um boxplot para facilitar a interpretação dos resultados.
Salva os resultados em log_transferencias.csv.
Envia o relatório automaticamente para o e-mail do destinatário informado.

🤖 Treinamento da Inteligência Artificial
Os dados históricos são lidos de arquivos CSV contendo registros validados como fraudes e não fraudes.
O modelo é treinado com regressão logística (ou outro modelo, como random forest).
A base é balanceada com SMOTE, para tratar o desbalanceamento entre classes.

🚀 Aplicações Reais
Fintechs e bancos digitais que desejam detectar e bloquear operações suspeitas.
Plataformas de gestão de risco ou monitoramento de fraudes.

✅ Conclusão
Este projeto fornece uma solução simples, automatizada e visual para a detecção de fraudes bancárias, com integração a relatórios e envio por e-mail. É ideal para uso acadêmico, protótipos de fintechs e sistemas de análise de risco.