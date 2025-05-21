# prever_fraude.py

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import smtplib
from email.message import EmailMessage
import os

def carregar_modelo_e_scaler():
    with open('modelo_fraude.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('scaler_fraude.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return modelo, scaler

def preprocessar_dados(novas_transferencias):
    novas_transferencias['valor_transferencia'] = novas_transferencias['valor_transferencia'].astype(str)
    novas_transferencias['valor_transferencia'] = novas_transferencias['valor_transferencia'].str.replace('.', '', regex=False)
    novas_transferencias['valor_transferencia'] = novas_transferencias['valor_transferencia'].str.replace(',', '.', regex=False)
    novas_transferencias['valor_transferencia'] = pd.to_numeric(novas_transferencias['valor_transferencia'], errors='coerce')
    novas_transferencias = novas_transferencias[novas_transferencias['valor_transferencia'] > 0]

    novas_transferencias['IP_origem_tranferencia'] = novas_transferencias['IP_origem_tranferencia'].astype(str).str.replace(r'\D', '', regex=True)
    novas_transferencias['IP_origem_tranferencia'] = pd.to_numeric(novas_transferencias['IP_origem_tranferencia'], errors='coerce')

    return novas_transferencias

def prever_transferencias(novas_transferencias, modelo, scaler):
    X_novas = novas_transferencias[['valor_transferencia', 'IP_origem_tranferencia']]
    X_novas_scaled = scaler.transform(X_novas)

    probas = modelo.predict_proba(X_novas_scaled)[:, 1]
    limiar = 0.02
    predicoes = ['Fraude' if p >= limiar else 'Não Fraude' for p in probas]
    novas_transferencias['fraude_prevista'] = predicoes
    return novas_transferencias

def coletar_dados_log_transferencias(df, caminho_csv):
    df_log = pd.DataFrame({
        'id_transferencia': df['Id_tranferencia'],
        'data_transferencia': df['Data_transferencia'],
        'valor_transferencia': df['valor_transferencia'],
        'id_fez_transferencia': df['id_fez_tranferencia'],
        'id_recebeu_transferencia': df['id_recebeu_transferencia'],
        'IP_origem_transferencia': df['IP_origem_tranferencia'],
        'tipo_transferencia': df['tipo_transferencia'],
        'status_transferencia': df['fraude_prevista'].apply(lambda x: 'Rejeitada' if x == 'Fraude' else 'Concluída')
    })

    df_log.to_csv(caminho_csv, index=False, encoding='utf-8-sig')
    print(f"\nArquivo de log gerado: {caminho_csv}")

def enviar_email_com_csv(arquivo_csv, destinatario, remetente, senha_email, servidor='smtp.gmail.com', porta=587):
    msg = EmailMessage()
    msg['Subject'] = 'Relatório de Log de Transferências'
    msg['From'] = remetente
    msg['To'] = destinatario
    msg.set_content('Segue em anexo o relatório de transferências com status de fraude.')

    with open(arquivo_csv, 'rb') as f:
        dados_csv = f.read()
        nome_arquivo = os.path.basename(arquivo_csv)
        msg.add_attachment(dados_csv, maintype='application', subtype='octet-stream', filename=nome_arquivo)

    try:
        with smtplib.SMTP(servidor, porta) as smtp:
            smtp.starttls()
            smtp.login(remetente, senha_email)
            smtp.send_message(msg)
            print(f"E-mail enviado com sucesso para {destinatario}")
    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")

def main():
    novos_user = pd.read_csv('dados_analise/novos_usuarios.csv', encoding='latin1')
    novas_contas = pd.read_csv('dados_analise/novas_contas.csv', encoding='latin1')
    novas_transferencias = pd.read_csv('dados_analise/novas_transferencias.csv', encoding='latin1')

    modelo, scaler = carregar_modelo_e_scaler()
    novas_transferencias = preprocessar_dados(novas_transferencias)
    novas_transferencias = prever_transferencias(novas_transferencias, modelo, scaler)

    # Gerar log e CSV
    caminho_csv = 'dados_analise/log_transferencias.csv'
    coletar_dados_log_transferencias(novas_transferencias, caminho_csv)

    remetente = 'wenneralmeida9@gmail.com'
    senha_email = 'udecigblqvxhqbwf'
    destinatario = 'w.almeida0502@gmail.com'

    enviar_email_com_csv(caminho_csv, destinatario, remetente, senha_email)

    print("\nRelatório de previsão das transferências:")
    print(novas_transferencias[['id_fez_tranferencia', 'valor_transferencia', 'id_recebeu_transferencia', 'IP_origem_tranferencia', 'fraude_prevista']])

    contagem = novas_transferencias['fraude_prevista'].value_counts().reindex(['Fraude', 'Não Fraude'], fill_value=0)
    cores = ['crimson', 'seagreen']
    categorias = ['Fraude', 'Não Fraude']
    valores = [contagem['Fraude'], contagem['Não Fraude']]

    plt.figure(figsize=(6, 4))
    barras = plt.bar(categorias, valores, color=cores)

    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, altura + 1, str(int(altura)), ha='center', va='bottom')

    plt.title('Previsões de Fraude nas Novas Transferências')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
