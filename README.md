# 🚨 Sistema de Detecção de Fraudes em Transferências Bancárias
Este projeto utiliza técnicas de aprendizado de máquina e pré-processamento de dados para prever possíveis fraudes em transferências financeiras com base no valor transferido e saldo da conta.

🔎 Objetivo: Ajudar instituições financeiras e fintechs a identificarem comportamentos anômalos em transações, aumentando a segurança e evitando perdas.

# 🧠 Bibliotecas Utilizadas
pandas – Manipulação de dados tabulares.
scikit-learn – Treinamento e avaliação de modelos de machine learning (incluindo regressão logística, random forest, métricas etc.).
imbalanced-learn (imblearn) – Balanceamento de classes usando a técnica SMOTE.
matplotlib – Visualização gráfica dos resultados, como curvas ROC e gráficos de barras.
seaborn – Estilização e gráficos estatísticos adicionais.
pickle – Serialização (salvar e carregar modelos), já incluído na biblioteca padrão do Python (não requer instalação via pip).

# 📦 Instalação das Dependências
  pip install pandas scikit-learn imbalanced-learn matplotlib

# 📁 Estrutura do código
├── dados_analise/
| |__ log_transferencias.csv (gerado após a previsão)
│ ├── novos_usuarios.csv
│ ├── novas_contas.csv
│ └── novas_transferencias.csv
├── modelo_fraude.pkl
├── scaler_fraude.pkl
└── treinar_modelo.py
└── prever_fraude_2.py


# 📌 Aplicações Reais
Este sistema pode ser utilizado por fintechs como Alelo, bancos digitais e sistemas de análise de risco para identificar transações suspeitas de forma automática e escalável.

# 📄 Coleta de Dados e Análise
Este sistema realiza a coleta de logs a partir de planilhas CSV, que contêm informações de usuários, contas e transferências bancárias. Esses dados são essenciais para realizar análises detalhadas e identificar possíveis indícios de fraude em transferências realizadas pelos usuários.

# 🧠 Treinamento da Inteligência Artificial
O treinamento da IA é feito com dados históricos armazenados em arquivos CSV. Esses dados contêm transferências previamente validadas como fraudes ou não fraudes. A partir disso, o sistema aprende a reconhecer padrões comportamentais suspeitos, utilizando técnicas de aprendizado supervisionado e balanceamento de dados (SMOTE).
Com esse treinamento, o modelo é capaz de prever se novas transferências são seguras ou possivelmente fraudulentas com base no valor e no saldo disponível do usuário.

#   📊 Geração de Gráficos e Relatórios
Após a análise das transferências, o sistema gera relatórios detalhados com as previsões realizadas pela IA, classificando cada operação como "Fraude" ou "Não Fraude".

Além disso, é criado um gráfico de barras que mostra visualmente a quantidade de transferências consideradas fraudulentas e não fraudulentas, facilitando a interpretação dos resultados por parte de analistas e gestores.

Essa funcionalidade é útil para monitoramento em tempo real, auditorias internas ou integração com plataformas de Business Intelligence (BI).

# ✅ Conclusão
Este sistema oferece uma solução prática e eficiente para detecção de fraudes em transferências bancárias, utilizando aprendizado de máquina supervisionado e dados reais em formato CSV.

Com um fluxo automatizado de coleta, pré-processamento, treinamento e previsão, o projeto permite identificar comportamentos suspeitos de forma simples e visual, podendo ser facilmente integrado a sistemas de monitoramento de instituições financeiras ou fintechs.#   M o s t r a _ t e c  
 