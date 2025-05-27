import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from prever_fraude_2 import (
    carregar_modelo_e_scaler,
    preprocessar_dados,
    prever_transferencias,
    coletar_dados_log_transferencias,
    enviar_email_com_csv
)

ARQUIVO_USUARIOS = 'dados_analise/novos_usuarios.csv'
ARQUIVO_CONTAS = 'dados_analise/novas_contas.csv'
ARQUIVO_TRANSFERENCIAS = 'dados_analise/novas_transferencias.csv'
CAMINHO_CSV_LOG = 'dados_analise/log_transferencias.csv'
EMAIL_REMETENTE = 'wenneralmeida9@gmail.com'
SENHA_EMAIL = 'udecigblqvxhqbwf'

class AppFraude:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detecção de Fraudes")
        self.root.geometry("1100x650")

        # Campo para e-mail destinatário
        self.email_label = tk.Label(root, text="E-mail do Destinatário:")
        self.email_label.pack(pady=5)
        self.email_entry = tk.Entry(root, width=40)
        self.email_entry.pack(pady=5)

        # Botões para previsão e envio
        self.btn_prever = tk.Button(root, text="Prever Fraudes", command=self.executar_previsao)
        self.btn_prever.pack(pady=10)

        self.btn_enviar = tk.Button(root, text="Enviar Relatório por E-mail", command=self.enviar_email)
        self.btn_enviar.pack(pady=5)

        # Tabela para mostrar resultados
        self.tree = ttk.Treeview(root, columns=("id", "valor", "destino", "ip", "fraude"), show="headings")
        self.tree.heading("id", text="Quem Transferiu")
        self.tree.heading("valor", text="Valor")
        self.tree.heading("destino", text="Quem Recebeu")
        self.tree.heading("ip", text="IP Origem")
        self.tree.heading("fraude", text="Status Previsto")
        self.tree.pack(expand=True, fill="both", pady=10)

        # Label de status
        self.status_label = tk.Label(root, text="")
        self.status_label.pack(pady=5)

        # Frame para os gráficos lado a lado
        self.frame_graficos = tk.Frame(root)
        self.frame_graficos.pack(expand=True, fill="both", pady=10)

        # Dois frames internos para cada gráfico
        self.frame_grafico1 = tk.Frame(self.frame_graficos)
        self.frame_grafico1.pack(side='left', expand=True, fill='both', padx=10)

        self.frame_grafico2 = tk.Frame(self.frame_graficos)
        self.frame_grafico2.pack(side='left', expand=True, fill='both', padx=10)

        self.df_resultado = None

    def executar_previsao(self):
        try:
            # Carrega os dados fixos
            novos_users = pd.read_csv(ARQUIVO_USUARIOS, encoding='latin1')
            novas_contas = pd.read_csv(ARQUIVO_CONTAS, encoding='latin1')
            novas_transf = pd.read_csv(ARQUIVO_TRANSFERENCIAS, encoding='latin1')

            # Carrega modelo e scaler
            modelo, scaler = carregar_modelo_e_scaler()

            # Processa e prevê fraudes
            novas_transf = preprocessar_dados(novas_transf)
            novas_transf = prever_transferencias(novas_transf, modelo, scaler)

            self.df_resultado = novas_transf

            coletar_dados_log_transferencias(novas_transf, CAMINHO_CSV_LOG)

            # Atualiza tabela na interface
            for item in self.tree.get_children():
                self.tree.delete(item)

            for _, row in novas_transf.iterrows():
                self.tree.insert("", "end", values=(
                    row['id_fez_tranferencia'],
                    row['valor_transferencia'],
                    row['id_recebeu_transferencia'],
                    row['IP_origem_tranferencia'],
                    row['fraude_prevista']
                ))

            self.status_label.config(text="Previsão realizada com sucesso!")

            # Exibe os dois gráficos
            self.exibir_graficos(novas_transf)

        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def enviar_email(self):
        email_destino = self.email_entry.get()
        if not email_destino:
            messagebox.showwarning("Atenção", "Digite o e-mail do destinatário.")
            return

        try:
            enviar_email_com_csv(CAMINHO_CSV_LOG, email_destino, EMAIL_REMETENTE, SENHA_EMAIL)
            messagebox.showinfo("Sucesso", f"E-mail enviado para {email_destino}")
        except Exception as e:
            messagebox.showerror("Erro ao Enviar E-mail", str(e))

    def exibir_graficos(self, df):
        # Limpa frames para novos gráficos
        for widget in self.frame_grafico1.winfo_children():
            widget.destroy()
        for widget in self.frame_grafico2.winfo_children():
            widget.destroy()

        # --- Gráfico 1: Contagem de Fraude e Não Fraude ---
        contagem = df['fraude_prevista'].value_counts().reindex(['Fraude', 'Não Fraude'], fill_value=0)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        barras = ax1.bar(['Fraude', 'Não Fraude'], [contagem['Fraude'], contagem['Não Fraude']],
                         color=['crimson', 'seagreen'])

        for barra in barras:
            altura = barra.get_height()
            ax1.text(barra.get_x() + barra.get_width() / 2, altura + 1, str(int(altura)),
                     ha='center', va='bottom', fontsize=12)

        ax1.set_title('Previsão de Fraudes', fontsize=14)
        ax1.set_ylabel('Quantidade', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        fig1.tight_layout()

        canvas1 = FigureCanvasTkAgg(fig1, master=self.frame_grafico1)
        canvas1.draw_idle()
        canvas1.get_tk_widget().pack(fill='both', expand=True)

        # --- Gráfico 2: Histograma dos Valores das Transferências ---
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.hist(df['valor_transferencia'], bins=30, color='dodgerblue', edgecolor='black')
        ax2.set_title('Distribuição dos Valores de Transferência', fontsize=14)
        ax2.set_xlabel('Valor da Transferência', fontsize=12)
        ax2.set_ylabel('Frequência', fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        fig2.tight_layout()
        
        canvas2 = FigureCanvasTkAgg(fig2, master=self.frame_grafico2)
        canvas2.draw_idle()
        canvas2.get_tk_widget().pack(fill='both', expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = AppFraude(root)
    root.mainloop()
