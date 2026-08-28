import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime, date

# Configuração da Página
st.set_page_config(
    page_title="Gestor de Pesquisa de Mestrado - PPGCC/UFSC",
    page_icon="🎓",
    layout="wide"
)

DB_PATH = "mestrado_tracker.db"

# --- BANCO DE DADOS & PERSISTÊNCIA ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS etapas (
            id INTEGER PRIMARY KEY,
            etapa_num INTEGER,
            nome TEXT,
            data_inicio TEXT,
            data_fim TEXT,
            progresso INTEGER,
            status TEXT,
            entregas TEXT,
            notas TEXT
        )
    """)
    conn.commit()
    
    # Inserir dados padrão se a tabela estiver vazia
    c.execute("SELECT COUNT(*) FROM etapas")
    if c.fetchone()[0] == 0:
        etapas_iniciais = [
            (1, 1, "Etapa 1: RSL, Alinhamento Metodológico e Disciplinas", "2026-08-31", "2026-12-31", 0, "Em Andamento", "Relatório de RSL + Créditos PPGCC", "Mapeamento das janelas de 90 dias e buscas no Scopus/IEEE."),
            (2, 2, "Etapa 2: Pipeline de Ingestão e Curadoria (ANA/INMET/MapBiomas)", "2027-01-01", "2027-04-30", 0, "Planejada", "Datasets curados em Parquet/DuckDB + Scripts de pré-processamento", "Normalização UTC, imputação e partição temporal sem leakage."),
            (3, 3, "Etapa 3: Baselines Tabulares e Redes Recorrentes (LSTM/GRU)", "2027-05-01", "2027-08-31", 0, "Planejada", "Relatório Parcial de Experimentos + Repositório Git", "Otimização via Optuna e métricas RMSE, MAE, NSE e Log-NSE."),
            (4, 4, "Etapa 4: Modelos de Atenção (TFT/Informer), Incerteza e Qualificação", "2027-09-01", "2027-12-31", 0, "Planejada", "Aprovação no Exame de Qualificação + Relatório Parcial FAPESC", "Regressão Quantílica (q=0.1, 0.5, 0.9) e detector ADWIN."),
            (5, 5, "Etapa 5: Benchmark Global, Análise de Robustez e Artigo Qualis A", "2028-01-01", "2028-04-30", 0, "Planejada", "Manuscrito de artigo submetido a periódico/conferência Qualis A", "Métricas CRPS, KGE e testes de significância estatística."),
            (6, 6, "Etapa 6: Redação Final, Defesa Pública e Consolidação Final", "2028-05-01", "2028-06-30", 0, "Planejada", "Dissertação homologada + Relatório Final FAPESC", "Defesa perante banca em Junho/2028 e depósito institucional.")
        ]
        c.executemany("""
            INSERT INTO etapas (id, etapa_num, nome, data_inicio, data_fim, progresso, status, entregas, notas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, etapas_iniciais)
        conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM etapas ORDER BY etapa_num ASC", conn)
    conn.close()
    return df

def update_etapa(etapa_id, progresso, status, notas):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE etapas 
        SET progresso = ?, status = ?, notas = ?
        WHERE id = ?
    """, (progresso, status, notas, etapa_id))
    conn.commit()
    conn.close()

init_db()
df_etapas = load_data()

# --- HEADER DA APLICAÇÃO ---
st.title("🎓 Painel de Acompanhamento da Pesquisa de Mestrado")
st.markdown("""
**Programa:** PPGCC / INE / CTC / UFSC
**Projeto:** *Modelagem Preditiva e Aprendizado Profundo para Séries Temporais Hidrológicas: Previsão de Vazão e Suporte à Gestão de Recursos Hídricos em Santa Catarina*
""")

st.divider()

# --- CARDS DE STATUS / KPIS ---
progresso_geral = int(df_etapas["progresso"].mean())
concluidas = len(df_etapas[df_etapas["status"] == "Concluída"])
em_andamento = len(df_etapas[df_etapas["status"] == "Em Andamento"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Progresso Geral", f"{progresso_geral}%")
col2.metric("Etapas em Andamento", f"{em_andamento}")
col3.metric("Etapas Concluídas", f"{concluidas} de {len(df_etapas)}")
col4.metric("Previsão de Defesa", "Junho / 2028")

st.progress(progresso_geral / 100)

# --- VISUALIZAÇÃO GANTT TIMELINE ---
st.subheader("📅 Cronograma Interativo (Linha do Tempo 2026 - 2028)")

df_plot = df_etapas.copy()
df_plot["data_inicio"] = pd.to_datetime(df_plot["data_inicio"])
df_plot["data_fim"] = pd.to_datetime(df_plot["data_fim"])

fig = px.timeline(
    df_plot,
    x_start="data_inicio",
    x_end="data_fim",
    y="nome",
    color="status",
    color_discrete_map={
        "Concluída": "#16a34a",
        "Em Andamento": "#2563eb",
        "Planejada": "#94a3b8",
        "Em Atraso": "#dc2626"
    },
    hover_data={"progresso": True, "entregas": True, "data_inicio": False, "data_fim": False},
    height=360
)
fig.update_yaxes(autorange="reversed", title="")
fig.update_xaxes(title="Período de Execução", dtick="M2", tickformat="%b/%Y")
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Status")
st.plotly_chart(fig, use_container_width=True)

# --- ATUALIZAÇÃO E DETALHAMENTO DAS ETAPAS ---
st.subheader("📝 Gerenciamento e Atualização de Tarefas")

tabs = st.tabs([f"Etapa {row['etapa_num']}" for _, row in df_etapas.iterrows()])

for i, (_, row) in enumerate(df_etapas.iterrows()):
    with tabs[i]:
        st.markdown(f"### {row['nome']}")
        c_left, c_right = st.columns([1.2, 1])
        
        with c_left:
            st.info(f"**Período:** {pd.to_datetime(row['data_inicio']).strftime('%d/%m/%Y')} até {pd.to_datetime(row['data_fim']).strftime('%d/%m/%Y')}")
            st.markdown(f"**📦 Entregas Formais Associadas:**\n- {row['entregas']}")
            
            with st.form(key=f"form_etapa_{row['id']}"):
                novo_status = st.selectbox(
                    "Status Atual:",
                    ["Planejada", "Em Andamento", "Concluída", "Em Atraso"],
                    index=["Planejada", "Em Andamento", "Concluída", "Em Atraso"].index(row["status"])
                )
                novo_progresso = st.slider("Percentual de Conclusão (%):", 0, 100, int(row["progresso"]), step=5)
                novas_notas = st.text_area("Registro de Atividades / Anotações do Laboratório:", value=row["notas"], height=100)
                
                submitted = st.form_submit_button("Salvar Atualizações")
                if submitted:
                    update_etapa(row["id"], novo_progresso, novo_status, novas_notas)
                    st.success("Progresso salvo com sucesso!")
                    st.rerun()

        with c_right:
            st.markdown("#### Detalhes Operacionais")
            if row["etapa_num"] == 1:
                st.markdown("""
                - [ ] RSL em bases indexadas (Scopus, IEEE Xplore, ACM).
                - [ ] Matrícula e cumprimento de disciplinas no PPGCC/UFSC.
                - [ ] Definição do tamanho da janela de lookback (90 dias).
                """)
            elif row["etapa_num"] == 2:
                st.markdown("""
                - [ ] Conector API ANA (SOAP/REST) e INMET (JSON).
                - [ ] Alinhamento UTC e agregação horária/diária.
                - [ ] Pipeline DuckDB/Parquet sem vazamento temporal.
                """)
            elif row["etapa_num"] == 3:
                st.markdown("""
                - [ ] Baselines tabulares (XGBoost / LightGBM) via Optuna.
                - [ ] Redes recorrentes (LSTM / GRU) em PyTorch.
                - [ ] Avaliação por métricas hidrológicas (RMSE, MAE, NSE, Log-NSE).
                """)
            elif row["etapa_num"] == 4:
                st.markdown("""
                - [ ] Modelos de Atenção (Temporal Fusion Transformer - TFT / Informer).
                - [ ] Regressão Quantílica com perda Pinball ($q=0.1, 0.5, 0.9$).
                - [ ] Mecanismo adaptativo de drift (ADWIN).
                - [ ] **Exame de Qualificação do Mestrado**.
                """)
            elif row["etapa_num"] == 5:
                st.markdown("""
                - [ ] Benchmark experimental global (CRPS, Pinball Loss, KGE).
                - [ ] Testes estatísticos de significância (Wilcoxon / Bootstrap).
                - [ ] Redação e submissão de artigo completo para periódico Qualis A.
                """)
            elif row["etapa_num"] == 6:
                st.markdown("""
                - [ ] Redação final dos capítulos no padrão BU/UFSC.
                - [ ] **Defesa Pública perante a Banca Examinadora (Jun/28)**.
                - [ ] Depósito institucional e Relatório Final FAPESC.
                """)