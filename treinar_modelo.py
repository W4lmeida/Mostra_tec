# treinar_modelo.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt

def carregar_dados(usuario_arquivo, conta_arquivo, transferencia_arquivo):
    usuarios = pd.read_csv(usuario_arquivo, encoding='latin1')
    contas = pd.read_csv(conta_arquivo, encoding='latin1')
    transferencias = pd.read_csv(transferencia_arquivo, encoding='latin1')
    return usuarios, contas, transferencias

def preprocesamento_dados(transferencias):
    transferencias['valor_transferencia'] = transferencias['valor_transferencia'].astype(str)
    transferencias['valor_transferencia'] = transferencias['valor_transferencia'].str.replace(',', '.', regex=False)
    transferencias['valor_transferencia'] = pd.to_numeric(transferencias['valor_transferencia'], errors='coerce')
    transferencias['IP_origem_tranferencia'] = transferencias['IP_origem_tranferencia'].astype(str).str.replace(r'\D', '', regex=True)
    transferencias['IP_origem_tranferencia'] = pd.to_numeric(transferencias['IP_origem_tranferencia'], errors='coerce')
    return transferencias

def treinando_modelo(transferencias):
    X = transferencias[['valor_transferencia', 'IP_origem_tranferencia']]
    y = transferencias['validacao'].apply(lambda x: 1 if x == 'Fraude' else 0)

    if len(y.value_counts()) < 2:
        raise ValueError("O conjunto de dados não possui ambas as classes ('Fraude' e 'Não Fraude').")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    modelo = RandomForestClassifier(random_state=42)

    y_real, y_probs = [], []

    for train_index, test_index in skf.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        modelo.fit(X_train, y_train)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        y_real.extend(y_test)
        y_probs.extend(y_proba)

    limiar = 0.5
    y_pred = [1 if p >= limiar else 0 for p in y_probs]

    print(f"\nAcurácia do modelo: {accuracy_score(y_real, y_pred):.2%}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_real, y_pred, target_names=['Não Fraude', 'Fraude']))

    fpr, tpr, _ = roc_curve(y_real, y_probs)
    auc_score = roc_auc_score(y_real, y_probs)

    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    with open('modelo_fraude.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    with open('scaler_fraude.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def main():
    usuarios, contas, transferencias = carregar_dados(
        'dados_treinamento/usuario_arquivo.csv',
        'dados_treinamento/conta_arquivo.csv',
        'dados_treinamento/transferencia_arquivo.csv'
    )
    transferencias = preprocesamento_dados(transferencias)
    treinando_modelo(transferencias)

if __name__ == "__main__":
    main()
