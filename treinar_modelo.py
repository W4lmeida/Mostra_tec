# Importamos ferramentas que vamos usar:
import pandas as pd  #organiza os dados da planilha
from sklearn.model_selection import StratifiedKFold  # Separa os dados para treinar e testar
from sklearn.preprocessing import StandardScaler  # Deixa os dados em uma mesma escala (padrão)
from sklearn.ensemble import RandomForestClassifier  # É a IA " que aprende a identificar fraudes
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve  # Avalia se o modelo está funcionando bem
from imblearn.over_sampling import SMOTE  # Ajuda a equilibrar casos de fraude e não-fraude
import pickle  # Serve para guardar o modelo que foi treinado
import matplotlib.pyplot as plt  # Cria gráficos

# Função que lê os arquivos com os dados
def carregar_dados(usuario_arquivo, conta_arquivo, transferencia_arquivo):
    # Lê os três arquivos: um com dados de usuários, outro com contas e outro com transferências
    usuarios = pd.read_csv(usuario_arquivo, encoding='latin1')
    contas = pd.read_csv(conta_arquivo, encoding='latin1')
    transferencias = pd.read_csv(transferencia_arquivo, encoding='latin1')
    return usuarios, contas, transferencias

# Função que limpa e prepara os dados de transferência
def preprocesamento_dados(transferencias):
    # Garante que o valor da transferência seja um número (pode fazer contas com ele)
    transferencias['valor_transferencia'] = transferencias['valor_transferencia'].astype(str)
    transferencias['valor_transferencia'] = transferencias['valor_transferencia'].str.replace(',', '.', regex=False)
    transferencias['valor_transferencia'] = pd.to_numeric(transferencias['valor_transferencia'], errors='coerce')

    # Remove letras e símbolos dos endereços IP e transforma em números
    transferencias['IP_origem_tranferencia'] = transferencias['IP_origem_tranferencia'].astype(str).str.replace(r'\D', '', regex=True)
    transferencias['IP_origem_tranferencia'] = pd.to_numeric(transferencias['IP_origem_tranferencia'], errors='coerce')

    return transferencias

# Função que treina o modelo para descobrir se uma transferência é fraude ou não
def treinando_modelo(transferencias):
    # Escolhemos quais dados a IA vai usar para aprender (valor e IP de origem)
    X = transferencias[['valor_transferencia', 'IP_origem_tranferencia']]

    # Dizemos o que queremos que o modelo aprenda: 1 = fraude, 0 = normal
    y = transferencias['validacao'].apply(lambda x: 1 if x == 'Fraude' else 0)

    # Verifica se tem exemplos de fraude e não fraude (se só tiver um tipo, não dá para aprender)
    if len(y.value_counts()) < 2:
        raise ValueError("É preciso ter exemplos de fraude e de transações normais.")

    # Padroniza os dados para ficarem todos na mesma base de comparação
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Equilibra os dados, criando exemplos de fraude se tiver poucos
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Separa os dados em 5 partes para testar várias vezes (isso melhora a confiança no resultado)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Criamos o "cérebro" que vai aprender (modelo de floresta aleatória)
    modelo = RandomForestClassifier(random_state=42)

    # Listas para guardar os resultados de verdade e os que o modelo chutou
    y_real, y_probs = [], []

    # Vamos treinar e testar o modelo 5 vezes, com partes diferentes dos dados
    for train_index, test_index in skf.split(X_resampled, y_resampled):
        # Separa uma parte para treino e outra para teste
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        # Treina o modelo com os dados de treino
        modelo.fit(X_train, y_train)

        # O modelo tenta adivinhar a chance de ser fraude em cada caso de teste
        y_proba = modelo.predict_proba(X_test)[:, 1]

        # Guarda os resultados verdadeiros e as apostas do modelo
        y_real.extend(y_test)
        y_probs.extend(y_proba)

    # Definimos um limite: se a chance de fraude for maior que 50%, é fraude
    limiar = 0.5
    y_pred = [1 if p >= limiar else 0 for p in y_probs]

    # Mostra na tela o quão certo o modelo está (porcentagem de acertos)
    print(f"\nAcurácia do modelo: {accuracy_score(y_real, y_pred):.2%}")

    # Mostra mais detalhes: quantas fraudes ele pegou, quantas ele errou etc.
    print("\nRelatório de Classificação:")
    print(classification_report(y_real, y_pred, target_names=['Não Fraude', 'Fraude']))

    # Desenha um gráfico que mostra como o modelo se saiu nos testes
    fpr, tpr, _ = roc_curve(y_real, y_probs)
    auc_score = roc_auc_score(y_real, y_probs)

    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Linha de referência (modelo aleatório)
    plt.xlabel('Falsos positivos (erros que achou que era fraude)')
    plt.ylabel('Verdades positivas (acertos de fraude)')
    plt.title('Gráfico de desempenho (Curva ROC)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # Guarda o modelo treinado e a ferramenta de padronização para usar depois
    with open('modelo_fraude.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    with open('scaler_fraude.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# Esta é a função principal que junta tudo e roda o programa
def main():
    # Carrega os dados de 3 arquivos diferentes
    usuarios, contas, transferencias = carregar_dados(
        'dados_treinamento/usuario_arquivo.csv',
        'dados_treinamento/conta_arquivo.csv',
        'dados_treinamento/transferencia_arquivo.csv'
    )

    # Prepara os dados das transferências (limpa e transforma)
    transferencias = preprocesamento_dados(transferencias)

    # Treina o modelo para detectar fraudes
    treinando_modelo(transferencias)

# Esta parte faz o programa rodar se você clicar "executar"
if __name__ == "__main__":
    main()
