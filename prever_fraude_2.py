# prever_fraude.py

# Importamos bibliotecas para:
# - lidar com planilhas (pandas),
# - criar gráficos (matplotlib),
# - carregar o modelo já treinado (pickle),
# - enviar e-mails (smtplib),
# - e manipular arquivos (os).
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import smtplib
from email.message import EmailMessage
import os
import seaborn as sns

# Carrega o modelo que foi treinado anteriormente, junto com o "ajustador" dos dados (scaler)
def carregar_modelo_e_scaler():
    with open('modelo_fraude.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('scaler_fraude.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return modelo, scaler

# Faz a limpeza e preparação dos dados da nova planilha de transferências
def preprocessar_dados(novas_transferencias):
    # Corrige valores com vírgula e ponto, e converte para número
    novas_transferencias['valor_transferencia'] = novas_transferencias['valor_transferencia'].astype(str)
    novas_transferencias['valor_transferencia'] = novas_transferencias['valor_transferencia'].str.replace('.', '', regex=False)
    novas_transferencias['valor_transferencia'] = novas_transferencias['valor_transferencia'].str.replace(',', '.', regex=False)
    novas_transferencias['valor_transferencia'] = pd.to_numeric(novas_transferencias['valor_transferencia'], errors='coerce')
    
    # Remove transferências com valor inválido (zero ou negativo)
    novas_transferencias = novas_transferencias[novas_transferencias['valor_transferencia'] > 0]

    # Remove símbolos do IP e transforma em número
    novas_transferencias['IP_origem_tranferencia'] = novas_transferencias['IP_origem_tranferencia'].astype(str).str.replace(r'\D', '', regex=True)
    novas_transferencias['IP_origem_tranferencia'] = pd.to_numeric(novas_transferencias['IP_origem_tranferencia'], errors='coerce')

    return novas_transferencias

# Utiliza o modelo carregado para prever se cada nova transferência é fraude ou não
def prever_transferencias(novas_transferencias, modelo, scaler):
    # Seleciona apenas os dados que o modelo precisa
    X_novas = novas_transferencias[['valor_transferencia', 'IP_origem_tranferencia']]

    # Aplica a mesma transformação usada no treinamento (padronização)
    X_novas_scaled = scaler.transform(X_novas)

    # O modelo calcula a probabilidade de cada transferência ser fraude
    probas = modelo.predict_proba(X_novas_scaled)[:, 1]

    # Define um limite: se a chance for maior que 2%, será considerada fraude
    limiar = 0.02
    predicoes = ['Fraude' if p >= limiar else 'Não Fraude' for p in probas]

    # Adiciona a previsão na tabela original
    novas_transferencias['fraude_prevista'] = predicoes
    return novas_transferencias

# Cria uma nova planilha de log com as previsões feitas
def coletar_dados_log_transferencias(df, caminho_csv):
    # Cria uma tabela com os dados principais + o status previsto (fraude ou não)
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

    # Salva esse log como um arquivo CSV (planilha)
    df_log.to_csv(caminho_csv, index=False, encoding='utf-8-sig')
    print(f"\nArquivo de log gerado: {caminho_csv}")

# Envia o CSV por e-mail com autenticação e anexo
def enviar_email_com_csv(arquivo_csv, destinatario, remetente, senha_email, servidor='smtp.gmail.com', porta=587):
    # Prepara o e-mail
    msg = EmailMessage()
    msg['Subject'] = 'Relatório de Log de Transferências'
    msg['From'] = remetente
    msg['To'] = destinatario
    msg.set_content('Segue em anexo o relatório de transferências do dia.')

    # Lê o arquivo e anexa ao e-mail
    with open(arquivo_csv, 'rb') as f:
        dados_csv = f.read()
        nome_arquivo = os.path.basename(arquivo_csv)
        msg.add_attachment(dados_csv, maintype='application', subtype='octet-stream', filename=nome_arquivo)

    # Conecta ao servidor de e-mail e envia a mensagem
    try:
        with smtplib.SMTP(servidor, porta) as smtp:
            smtp.starttls()  # Garante segurança
            smtp.login(remetente, senha_email)
            smtp.send_message(msg)
            print(f"E-mail enviado com sucesso para {destinatario}")
    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")

# Função principal que executa todo o processo
def main():
    # Carrega as novas planilhas com dados de usuários, contas e transferências
    novos_user = pd.read_csv('dados_analise/novos_usuarios.csv', encoding='latin1')
    novas_contas = pd.read_csv('dados_analise/novas_contas.csv', encoding='latin1')
    novas_transferencias = pd.read_csv('dados_analise/novas_transferencias.csv', encoding='latin1')

    # Carrega o modelo de detecção de fraudes já treinado
    modelo, scaler = carregar_modelo_e_scaler()

    # Prepara os dados das transferências novas
    novas_transferencias = preprocessar_dados(novas_transferencias)

    # Realiza as previsões de fraude com o modelo
    novas_transferencias = prever_transferencias(novas_transferencias, modelo, scaler)

    # Gera um arquivo com os resultados em forma de log
    caminho_csv = 'dados_analise/log_transferencias.csv'
    coletar_dados_log_transferencias(novas_transferencias, caminho_csv)

    # Dados para envio do relatório por e-mail
    remetente = 'wenneralmeida9@gmail.com'
    senha_email = 'udecigblqvxhqbwf'
    destinatario = 'w.almeida0502@gmail.com'

    # Envia o arquivo CSV para o destinatário
    enviar_email_com_csv(caminho_csv, destinatario, remetente, senha_email)

    # Exibe no terminal um resumo com as transferências e suas classificações
    print("\nRelatório de previsão das transferências:")
    print(novas_transferencias[['id_fez_tranferencia', 'valor_transferencia', 'id_recebeu_transferencia', 'IP_origem_tranferencia', 'fraude_prevista']])

    # Conta quantas foram fraude e quantas não foram
    contagem = novas_transferencias['fraude_prevista'].value_counts().reindex(['Fraude', 'Não Fraude'], fill_value=0)

    # Cria um gráfico de barras para visualizar os resultados
    cores = ['crimson', 'seagreen']
    categorias = ['Fraude', 'Não Fraude']
    valores = [contagem['Fraude'], contagem['Não Fraude']]

    plt.figure(figsize=(6, 4))
    barras = plt.bar(categorias, valores, color=cores)

    # Adiciona os números em cima das barras
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, altura + 1, str(int(altura)), ha='center', va='bottom')

    # Configurações finais do gráfico
    plt.title('Previsões de Fraude nas Novas Transferências')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Gráfico 2: boxplot de valores das transferências
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='fraude_prevista', y='valor_transferencia', data=novas_transferencias, palette={'Fraude': 'crimson', 'Não Fraude': 'seagreen'})
    plt.title('Distribuição dos Valores das Transferências por Previsão')
    plt.xlabel('Previsão')
    plt.ylabel('Valor da Transferência')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Gráfico 3: gráfico de pizza com porcentagens
    plt.figure(figsize=(6, 6))
    plt.pie(valores, labels=categorias, colors=cores, autopct='%1.1f%%', startangle=140, explode=(0.1, 0))
    plt.title('Proporção de Fraudes e Não Fraudes nas Transferências')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Executa o programa se o arquivo for rodado diretamente
if __name__ == "__main__":
    main()
