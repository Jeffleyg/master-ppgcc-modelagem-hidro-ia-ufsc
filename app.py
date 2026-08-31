"""
HydraOps - Enterprise Research & Data Engineering Platform
PPGCC / INE / CTC / UFSC - Time Series ML & Hydrological Forecasting Hub
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json
import io

# ==========================================
# 1. PAGE CONFIG & ENTERPRISE CSS INJECTION
# ==========================================
st.set_page_config(
    page_title="HydraOps | PPGCC UFSC Research Hub",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Metric Cards Custom Styling */
    div[data-testid="stMetric"] {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 500;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        background-color: #1e293b;
        color: #cbd5e1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: bold;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DB_PATH = "mestrado_tracker.db"

# ==========================================
# 2. PERSISTENCE LAYER & DATABASE SETUP
# ==========================================
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Tabela de Etapas do Cronograma
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
            notas TEXT,
            updated_at TEXT
        )
    """)
    
    # Migração defensiva: garante a existência da coluna updated_at em tabelas antigas
    c.execute("PRAGMA table_info(etapas)")
    columns = [row[1] for row in c.fetchall()]
    if "updated_at" not in columns:
        c.execute("ALTER TABLE etapas ADD COLUMN updated_at TEXT")
        c.execute("UPDATE etapas SET updated_at = datetime('now') WHERE updated_at IS NULL")
    
    # 2. Tabela de Micro-Checklist Operacional
    c.execute("""
        CREATE TABLE IF NOT EXISTS checklist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etapa_id INTEGER,
            item TEXT,
            concluido INTEGER DEFAULT 0,
            FOREIGN KEY (etapa_id) REFERENCES etapas(id)
        )
    """)
    
    # 3. Tabela de Model Registry
    c.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_family TEXT,
            architecture TEXT,
            lookback_window INTEGER,
            horizon_steps INTEGER,
            rmse REAL,
            mae REAL,
            nse REAL,
            log_nse REAL,
            kge REAL,
            crps REAL,
            pinball_loss REAL,
            status TEXT,
            notes TEXT
        )
    """)
    
    # 4. Tabela de Telemetria do Data Lakehouse
    c.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source_name TEXT,
            layer TEXT,
            records_processed INTEGER,
            null_ratio REAL,
            drift_score REAL,
            adwin_alert INTEGER,
            latency_sec REAL,
            status TEXT
        )
    """)
    conn.commit()

    # Seed Tabela: etapas
    c.execute("SELECT COUNT(*) FROM etapas")
    if c.fetchone()[0] == 0:
        etapas_iniciais = [
            (1, 1, "Etapa 1: RSL, Alinhamento Metodológico e Créditos PPGCC", "2026-08-31", "2026-12-31", 15, "Em Andamento", "Relatório de RSL + Créditos PPGCC", "Mapeamento das janelas de 90 dias, buscas Scopus/IEEE e estado da arte."),
            (2, 2, "Etapa 2: Pipeline de Ingestão e Curadoria (ANA/INMET/MapBiomas)", "2027-01-01", "2027-04-30", 0, "Planejada", "Datasets curados em Parquet/DuckDB + Scripts de pré-processamento", "Normalização UTC, imputação e partição temporal sem leakage."),
            (3, 3, "Etapa 3: Baselines Tabulares e Redes Recorrentes (LSTM/GRU)", "2027-05-01", "2027-08-31", 0, "Planejada", "Relatório Parcial de Experimentos + Repositório Git", "Otimização via Optuna e métricas RMSE, MAE, NSE e Log-NSE."),
            (4, 4, "Etapa 4: Modelos de Atenção (TFT/Informer), Incerteza e Qualificação", "2027-09-01", "2027-12-31", 0, "Planejada", "Aprovação no Exame de Qualificação + Relatório Parcial FAPESC", "Regressão Quantílica (q=0.1, 0.5, 0.9) e detector adaptativo ADWIN."),
            (5, 5, "Etapa 5: Benchmark Global, Análise de Robustez e Artigo Qualis A", "2028-01-01", "2028-04-30", 0, "Planejada", "Manuscrito de artigo submetido a periódico/conferência Qualis A", "Métricas CRPS, KGE e testes de significância estatística (Wilcoxon/Bootstrap)."),
            (6, 6, "Etapa 6: Redação Final, Defesa Pública e Consolidação Final", "2028-05-01", "2028-06-30", 0, "Planejada", "Dissertação homologada + Relatório Final FAPESC", "Defesa perante banca em Junho/2028 e depósito institucional BU/UFSC.")
        ]
        c.executemany("""
            INSERT INTO etapas (id, etapa_num, nome, data_inicio, data_fim, progresso, status, entregas, notas, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, etapas_iniciais)
        conn.commit()

    # Seed Tabela: checklist
    c.execute("SELECT COUNT(*) FROM checklist")
    if c.fetchone()[0] == 0:
        checklist_items = [
            (1, "RSL em bases indexadas (Scopus, IEEE Xplore, ACM DL)", 1),
            (1, "Definição formal do problema & Lookback Window (90 dias)", 1),
            (1, "Cumprimento de 12 créditos em disciplinas PPGCC/UFSC", 0),
            (2, "Conector API ANA (Telemetria Hidroweb / SOAP & REST)", 0),
            (2, "Conector BDMEP / INMET (Estações Convencionais & Automáticas)", 0),
            (2, "Pipeline de Engenharia no DuckDB com persistência Snappy-Parquet", 0),
            (2, "Time-split rígido com purge/embargo anti-vazamento temporal", 0),
            (3, "Implementação de Baselines Tabulares (XGBoost, LightGBM, CatBoost)", 0),
            (3, "Arquiteturas Recorrentes em PyTorch (LSTM, BiLSTM, GRU)", 0),
            (3, "Pipeline de HPO via Optuna (Tree-structured Parzen Estimator)", 0),
            (4, "Arquiteturas Temporal Fusion Transformer (TFT) & Informer", 0),
            (4, "Loss Quantílica Pinball para quantificação de incerteza (P10, P50, P90)", 0),
            (4, "Detector de Drift Conceitual Hidrológico com ADWIN", 0),
            (4, "Banca do Exame de Qualificação PPGCC/UFSC", 0),
            (5, "Avaliação Multicritério (CRPS, KGE, NSE, Log-NSE)", 0),
            (5, "Testes de Hipótese Estatística (Wilcoxon Signed-Rank, Diebold-Mariano)", 0),
            (5, "Redação e submissão de paper para periódico Qualis A1/A2", 0),
            (6, "Redação integral dos capítulos da Dissertação (Template LaTeX PPGCC)", 0),
            (6, "Defesa Pública perante a Banca Examinadora (Jun/2028)", 0),
            (6, "Depósito Institucional na Biblioteca Universitária (BU/UFSC)", 0)
        ]
        c.executemany("INSERT INTO checklist (etapa_id, item, concluido) VALUES (?, ?, ?)", checklist_items)
        conn.commit()

    # Seed Tabela: model_runs
    c.execute("SELECT COUNT(*) FROM model_runs")
    if c.fetchone()[0] == 0:
        sample_runs = [
            ("2026-08-25 14:30:00", "Baseline", "Climatology Mean", 90, 7, 142.5, 98.2, 0.12, 0.05, 0.21, 84.1, 0.450, "Completed", "Benchmark ingênuo"),
            ("2026-08-26 10:15:00", "Tabular GBDT", "LightGBM Regressor", 90, 7, 78.4, 52.1, 0.74, 0.69, 0.78, 48.2, 0.260, "Completed", "Features com lags 1..90 + rolling stats"),
            ("2026-08-27 16:45:00", "Recurrent NN", "Stacked LSTM (2 layers)", 90, 7, 62.1, 41.5, 0.83, 0.79, 0.84, 39.0, 0.205, "Completed", "Hidden dim 128, Dropout 0.2"),
            ("2026-08-28 09:20:00", "Recurrent NN", "GRU + Attention Head", 90, 7, 56.8, 37.9, 0.87, 0.83, 0.88, 34.2, 0.180, "Completed", "Mecanismo Bahdanau de atenção"),
            ("2026-08-30 18:00:00", "Transformer", "Temporal Fusion Transformer (TFT)", 90, 7, 44.2, 29.3, 0.93, 0.90, 0.94, 25.1, 0.135, "Completed", "Variáveis exógenas: precipitação, evapotranspiração"),
        ]
        c.executemany("""
            INSERT INTO model_runs (timestamp, model_family, architecture, lookback_window, horizon_steps, rmse, mae, nse, log_nse, kge, crps, pinball_loss, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_runs)
        conn.commit()

    # Seed Tabela: pipeline_runs
    c.execute("SELECT COUNT(*) FROM pipeline_runs")
    if c.fetchone()[0] == 0:
        sample_pipelines = [
            ("2026-08-31 08:00:00", "ANA Telemetria", "Bronze -> Silver", 1420500, 0.012, 0.04, 0, 14.2, "SUCCESS"),
            ("2026-08-31 08:05:00", "INMET Estações", "Bronze -> Silver", 845000, 0.008, 0.02, 0, 8.7, "SUCCESS"),
            ("2026-08-31 08:12:00", "MapBiomas LULC", "Silver -> Gold", 312000, 0.000, 0.01, 0, 4.1, "SUCCESS"),
            ("2026-08-31 08:20:00", "Feature Store Hydro", "Gold (Training-Ready)", 2577500, 0.000, 0.05, 0, 22.5, "SUCCESS"),
        ]
        c.executemany("""
            INSERT INTO pipeline_runs (timestamp, source_name, layer, records_processed, null_ratio, drift_score, adwin_alert, latency_sec, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_pipelines)
        conn.commit()

init_db()

# ==========================================
# 3. DATA ACCESS LAYER & MUTATIONS
# ==========================================
def get_etapas_df():
    conn = get_db_connection()
    return pd.read_sql("SELECT * FROM etapas ORDER BY etapa_num ASC", conn)

def get_checklist_df(etapa_id=None):
    conn = get_db_connection()
    if etapa_id is not None:
        return pd.read_sql("SELECT * FROM checklist WHERE etapa_id = ? ORDER BY id ASC", conn, params=(etapa_id,))
    return pd.read_sql("SELECT * FROM checklist ORDER BY etapa_id, id ASC", conn)

def get_model_runs_df():
    conn = get_db_connection()
    return pd.read_sql("SELECT * FROM model_runs ORDER BY id DESC", conn)

def get_pipeline_runs_df():
    conn = get_db_connection()
    return pd.read_sql("SELECT * FROM pipeline_runs ORDER BY id DESC", conn)

def update_etapa_record(etapa_id, progresso, status, notas):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        UPDATE etapas 
        SET progresso = ?, status = ?, notas = ?, updated_at = datetime('now')
        WHERE id = ?
    """, (progresso, status, notas, etapa_id))
    conn.commit()

def toggle_checklist_item(item_id, current_state):
    conn = get_db_connection()
    c = conn.cursor()
    new_state = 0 if current_state == 1 else 1
    c.execute("UPDATE checklist SET concluido = ? WHERE id = ?", (new_state, item_id))
    
    # Recalcular progresso da etapa com base nas tarefas concluídas
    c.execute("SELECT etapa_id FROM checklist WHERE id = ?", (item_id,))
    row_etapa = c.fetchone()
    if row_etapa:
        etapa_id = row_etapa[0]
        c.execute("SELECT COUNT(*), SUM(concluido) FROM checklist WHERE etapa_id = ?", (etapa_id,))
        total, completed = c.fetchone()
        if total and total > 0:
            completed = completed or 0
            pct = int((completed / total) * 100)
            status = "Concluída" if pct == 100 else ("Em Andamento" if pct > 0 else "Planejada")
            c.execute("UPDATE etapas SET progresso = ?, status = ?, updated_at = datetime('now') WHERE id = ?", (pct, status, etapa_id))
    
    conn.commit()

def log_new_experiment(family, arch, lookback, horizon, rmse, mae, nse, log_nse, kge, crps, pinball, notes):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO model_runs (timestamp, model_family, architecture, lookback_window, horizon_steps, rmse, mae, nse, log_nse, kge, crps, pinball_loss, status, notes)
        VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Completed', ?)
    """, (family, arch, lookback, horizon, rmse, mae, nse, log_nse, kge, crps, pinball, notes))
    conn.commit()

def trigger_pipeline_sync(source, layer, records, nulls, drift, adwin, latency):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO pipeline_runs (timestamp, source_name, layer, records_processed, null_ratio, drift_score, adwin_alert, latency_sec, status)
        VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, 'SUCCESS')
    """, (source, layer, records, nulls, drift, adwin, latency))
    conn.commit()

# ==========================================
# 4. SIDEBAR NAVIGATION & HEALTH
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("## **HydraOps Platform**")
    st.caption("PPGCC / UFSC - Time-Series ML & Data Lakehouse Hub")
    
    st.markdown("---")
    navigation = st.radio(
        "Navegação do Hub",
        [
            "🏛️ Visão Executiva & Cronograma",
            "🛠️ Engenharia de Dados (Lakehouse)",
            "🧠 ML Registry & Benchmarking",
            "📊 Analytics & Métricas Hidrológicas",
            "⚙️ Centro de Operações & DB"
        ]
    )
    
    st.markdown("---")
    st.markdown("### **Status do Ambiente**")
    st.markdown("""
    - **Cluster Engine:** `DuckDB v1.0.0`
    - **Compute Target:** `PyTorch 2.4 (CUDA 12.4)`
    - **Telemetry Store:** `SQLite Engine`
    - **Storage Format:** `Apache Parquet (Snappy)`
    - **Orchestrator:** `Airflow DAGs Active`
    """)
    
    st.markdown("---")
    df_pipe = get_pipeline_runs_df()
    total_records = df_pipe["records_processed"].sum() if not df_pipe.empty else 0
    st.metric("Total de Registos Ingeridos", f"{total_records:,.0f}".replace(",", "."))

# ==========================================
# 5. MODULE 1: VISÃO EXECUTIVA & CRONOGRAMA
# ==========================================
if navigation == "🏛️ Visão Executiva & Cronograma":
    st.title("🎓 Painel de Gestão da Pesquisa de Mestrado")
    st.markdown("""
    **Programa de Pós-Graduação em Ciência da Computação (PPGCC / INE / CTC / UFSC)**  
    **Linha de Pesquisa:** Inteligência Computacional & Sistemas de Dados de Alta Performance  
    **Projeto:** *Modelagem Preditiva e Aprendizado Profundo para Séries Temporais Hidrológicas: Previsão de Vazão e Suporte à Gestão de Recursos Hídricos em Santa Catarina*
    """)
    
    st.markdown("---")
    
    df_etapas = get_etapas_df()
    df_chk = get_checklist_df()
    
    progresso_ponderado = int(df_etapas["progresso"].mean()) if not df_etapas.empty else 0
    concluidas = len(df_etapas[df_etapas["status"] == "Concluída"]) if not df_etapas.empty else 0
    em_andamento = len(df_etapas[df_etapas["status"] == "Em Andamento"]) if not df_etapas.empty else 0
    total_tarefas = len(df_chk)
    tarefas_concluidas = len(df_chk[df_chk["concluido"] == 1]) if not df_chk.empty else 0
    pct_tarefas = int((tarefas_concluidas / total_tarefas) * 100) if total_tarefas > 0 else 0
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Progresso do Mestrado", f"{progresso_ponderado}%", delta=f"{progresso_ponderado - 10}% vs Mês Ant.")
    kpi2.metric("Etapas em Execução", f"{em_andamento}", help="Etapas ativas no cronograma")
    kpi3.metric("Marcos Concluídos", f"{concluidas} / {len(df_etapas)}")
    kpi4.metric("Micro-Tarefas Entregues", f"{tarefas_concluidas} / {total_tarefas}", delta=f"{pct_tarefas}% concluído")
    kpi5.metric("Data Prevista da Defesa", "30/06/2028", delta="-668 dias", delta_color="off")
    
    st.progress(progresso_ponderado / 100)
    
    st.markdown("### 📅 Cronograma Interativo da Pesquisa (2026 - 2028)")
    
    if not df_etapas.empty:
        df_plot = df_etapas.copy()
        df_plot["data_inicio"] = pd.to_datetime(df_plot["data_inicio"])
        df_plot["data_fim"] = pd.to_datetime(df_plot["data_fim"])
        
        fig_gantt = px.timeline(
            df_plot,
            x_start="data_inicio",
            x_end="data_fim",
            y="nome",
            color="status",
            color_discrete_map={
                "Concluída": "#16a34a",
                "Em Andamento": "#2563eb",
                "Planejada": "#64748b",
                "Em Atraso": "#dc2626"
            },
            hover_data={"progresso": True, "entregas": True, "data_inicio": False, "data_fim": False},
            height=380
        )
        fig_gantt.update_yaxes(autorange="reversed", title="")
        fig_gantt.update_xaxes(title="Linha do Tempo de Execução", dtick="M3", tickformat="%b/%Y")
        fig_gantt.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gantt, width="stretch")
    
    st.markdown("### 📝 Gerenciamento Atómico por Macro-Etapa")
    if not df_etapas.empty:
        tabs = st.tabs([f"Etapa {row['etapa_num']}: {row['nome'].split(':')[1][:25]}..." for _, row in df_etapas.iterrows()])
        
        for i, (_, row) in enumerate(df_etapas.iterrows()):
            with tabs[i]:
                c_left, c_right = st.columns([1.3, 1.0])
                
                with c_left:
                    st.markdown(f"#### 🎯 {row['nome']}")
                    st.info(f"**Janela de Execução:** {pd.to_datetime(row['data_inicio']).strftime('%d/%m/%Y')} ➔ {pd.to_datetime(row['data_fim']).strftime('%d/%m/%Y')}")
                    st.markdown(f"**📦 Entregáveis Associados:**\n- {row['entregas']}")
                    
                    with st.form(key=f"form_etapa_{row['id']}"):
                        novo_status = st.selectbox(
                            "Status Institucional:",
                            ["Planejada", "Em Andamento", "Concluída", "Em Atraso"],
                            index=["Planejada", "Em Andamento", "Concluída", "Em Atraso"].index(row["status"])
                        )
                        novo_progresso = st.slider("Percentual Manual de Progresso (%):", 0, 100, int(row["progresso"]), step=5)
                        novas_notas = st.text_area("Diário de Bordo & Notas Técnicas do Lab:", value=row["notas"] if row["notas"] else "", height=90)
                        
                        if st.form_submit_button("💾 Atualizar Metadados da Etapa", width="stretch"):
                            update_etapa_record(row["id"], novo_progresso, novo_status, novas_notas)
                            st.success("Metadados sincronizados na base SQLite.")
                            st.rerun()

                with c_right:
                    st.markdown("#### ✅ Checklist Operacional Atômico")
                    st.caption("A alteração de itens recalcula automaticamente o progresso percentual da etapa.")
                    
                    items = get_checklist_df(row["id"])
                    for _, item in items.iterrows():
                        is_checked = bool(item["concluido"])
                        if st.checkbox(item["item"], value=is_checked, key=f"chk_{item['id']}"):
                            if not is_checked:
                                toggle_checklist_item(item["id"], item["concluido"])
                                st.rerun()
                        else:
                            if is_checked:
                                toggle_checklist_item(item["id"], item["concluido"])
                                st.rerun()

# ==========================================
# 6. MODULE 2: ENGENHARIA DE DADOS (LAKEHOUSE)
# ==========================================
elif navigation == "🛠️ Engenharia de Dados (Lakehouse)":
    st.title("🛠️ Arquitetura do Data Lakehouse Hidrológico")
    st.caption("Pipeline de Ingestão Medallion (Bronze ➔ Silver ➔ Gold) & Monitorização de Drift")
    
    st.markdown("---")
    
    col_arch1, col_arch2, col_arch3 = st.columns(3)
    with col_arch1:
        st.markdown(r"""
        #### 🥉 Camada Bronze (Raw Ingestion)
        - **ANA Telemetria:** Protocolos SOAP / REST HidroWeb.
        - **INMET BDMEP:** Estações automáticas & convencionais.
        - **MapBiomas:** Cobertura e uso do solo (Raster/GeoTIFF).
        - **Formato:** JSON Bruto / CSV particionado por bacia.
        """)
    with col_arch2:
        st.markdown(r"""
        #### 🥈 Camada Silver (Curated & Validated)
        - **Alinhamento Temporal:** UTC Standard & Resample horário/diário.
        - **Validação de Schema:** Pydantic & Great Expectations.
        - **Tratamento de Outliers:** Filtro Hampel + Imputação PCHIP.
        - **Formato:** Apache Parquet (Snappy) indexado por Estação ID.
        """)
    with col_arch3:
        st.markdown(r"""
        #### 🥇 Camada Gold (Feature Store ML)
        - **Lookback Windows:** Lag matrices ($t-1, \dots, t-90$).
        - **Rolling Statistics:** Médias móveis, EWMA, ETP Penman-Monteith.
        - **Purge & Embargo:** Partição temporal com zero data leakage.
        - **Storage:** DuckDB Analytics Store.
        """)
    
    st.markdown("---")
    st.subheader("⚡ Telemetria dos Pipelines ETL em Tempo Real")
    
    df_pipe = get_pipeline_runs_df()
    
    if not df_pipe.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Throughput de Registos", f"{df_pipe['records_processed'].sum():,.0f}".replace(",", "."))
        m2.metric("Taxa Média de Nulos", f"{(df_pipe['null_ratio'].mean()*100):.2f}%", delta="-0.04% vs Target < 1%")
        m3.metric("Média Latência de Ingestão", f"{df_pipe['latency_sec'].mean():.1f} seg")
        m4.metric("Alertas ADWIN Drift", f"{int(df_pipe['adwin_alert'].sum())}", delta="0 Críticos", delta_color="normal")
        
        st.dataframe(
            df_pipe.style.format({
                "records_processed": "{:,.0f}",
                "null_ratio": "{:.4f}",
                "drift_score": "{:.4f}",
                "latency_sec": "{:.2f} s"
            }),
            width="stretch"
        )
    
    with st.expander("➕ Disparar / Simular Ingestão Manual de Pipeline (ETL Trigger)"):
        with st.form("manual_etl_form"):
            c1, c2, c3 = st.columns(3)
            src = c1.selectbox("Data Source", ["ANA Telemetria", "INMET BDMEP", "MapBiomas LULC", "ERA5 Land Reanalysis", "Feature Store Hydro"])
            layer = c2.selectbox("Camada Destino", ["Bronze -> Silver", "Silver -> Gold", "Gold (Training-Ready)"])
            records = c3.number_input("Registos Processados", min_value=1000, max_value=10000000, value=250000, step=50000)
            
            c4, c5, c6 = st.columns(3)
            nulls = c4.number_input("Taxa de Nulos Estimada", min_value=0.0, max_value=1.0, value=0.002, format="%.4f")
            drift = c5.number_input("Score KS/Wasserstein Drift", min_value=0.0, max_value=1.0, value=0.03, format="%.4f")
            lat = c6.number_input("Tempo de Execução (s)", min_value=0.1, max_value=300.0, value=12.4)
            
            if st.form_submit_button("🚀 Executar Job de Ingestão"):
                trigger_pipeline_sync(src, layer, records, nulls, drift, 1 if drift > 0.15 else 0, lat)
                st.success("Pipeline executado com sucesso e metadados arquivados no Lakehouse Registry.")
                st.rerun()

# ==========================================
# 7. MODULE 3: ML REGISTRY & BENCHMARKING
# ==========================================
elif navigation == "🧠 ML Registry & Benchmarking":
    st.title("🧠 Model Registry & Benchmark Experimental")
    st.caption("Registo formal de experiências, scoring multicritério e rastreamento de hiperparâmetros (MLflow-like)")
    
    st.markdown("---")
    
    df_models = get_model_runs_df()
    
    st.markdown("### 🏆 Leaderboard de Modelos (Ordenado por NSE / Coeficiente de Nash-Sutcliffe)")
    
    if not df_models.empty:
        best_model = df_models.sort_values(by="nse", ascending=False).iloc[0]
        st.success(f"🥇 **Modelo Campeão Atual:** `{best_model['architecture']}` ({best_model['model_family']}) — **NSE:** `{best_model['nse']:.3f}` | **KGE:** `{best_model['kge']:.3f}` | **CRPS:** `{best_model['crps']:.2f}`")
        
        st.dataframe(
            df_models[[
                "id", "timestamp", "model_family", "architecture", "lookback_window", 
                "horizon_steps", "rmse", "mae", "nse", "log_nse", "kge", "crps", "status"
            ]].style.format({
                "rmse": "{:.2f}",
                "mae": "{:.2f}",
                "nse": "{:.3f}",
                "log_nse": "{:.3f}",
                "kge": "{:.3f}",
                "crps": "{:.2f}"
            }).background_gradient(subset=["nse", "kge"], cmap="Greens")
              .background_gradient(subset=["rmse", "mae", "crps"], cmap="Reds_r"),
            width="stretch"
        )
    
    st.markdown("---")
    st.subheader("🧪 Submeter Novo Experimento de Treino ao Registry")
    
    with st.form("new_experiment_form"):
        r1, r2, r3 = st.columns(3)
        family = r1.selectbox("Família do Modelo", ["Baseline", "Tabular GBDT", "Recurrent NN", "Transformer", "Hydrological Physical", "Hybrid Physics-Informed"])
        arch = r2.text_input("Nome da Arquitetura", value="Temporal Fusion Transformer (TFT)")
        lookback = r3.number_input("Janela de Lookback (dias)", min_value=7, max_value=365, value=90)
        
        r4, r5, r6 = st.columns(3)
        horizon = r4.number_input("Horizonte Preditivo (dias)", min_value=1, max_value=30, value=7)
        rmse_val = r5.number_input("RMSE (m³/s)", min_value=0.0, value=38.4, format="%.2f")
        mae_val = r6.number_input("MAE (m³/s)", min_value=0.0, value=24.1, format="%.2f")
        
        r7, r8, r9 = st.columns(3)
        nse_val = r7.number_input("NSE (Nash-Sutcliffe)", min_value=-10.0, max_value=1.0, value=0.945, format="%.3f")
        log_nse_val = r8.number_input("Log-NSE (Vazões de Estiagem)", min_value=-10.0, max_value=1.0, value=0.912, format="%.3f")
        kge_val = r9.number_input("KGE (Kling-Gupta Efficiency)", min_value=-10.0, max_value=1.0, value=0.952, format="%.3f")
        
        r10, r11 = st.columns(2)
        crps_val = r10.number_input("CRPS (Continuous Ranked Probability Score)", min_value=0.0, value=21.4, format="%.2f")
        pinball_val = r11.number_input("Pinball Loss (Média q=0.1, 0.5, 0.9)", min_value=0.0, value=0.115, format="%.4f")
        
        notes_val = st.text_area("Hiperparâmetros e Observações Experimentais", value="Otimizado via Optuna (200 trials). Variáveis exógenas: precipitação acumulada e índices de vegetação.")
        
        if st.form_submit_button("📥 Registar Experimento no Leaderboard"):
            log_new_experiment(family, arch, lookback, horizon, rmse_val, mae_val, nse_val, log_nse_val, kge_val, crps_val, pinball_val, notes_val)
            st.success("Experimento catalogado com sucesso.")
            st.rerun()

# ==========================================
# 8. MODULE 4: ANALYTICS & MÉTRICAS HIDROLÓGICAS
# ==========================================
elif navigation == "📊 Analytics & Métricas Hidrológicas":
    st.title("📊 Análise Multicritério & Simulação Hidrológica")
    st.caption("Comparação de Desempenho em Eventos Extremos (Cheias vs Estiagens) e Incerteza Preditiva")
    
    st.markdown("---")
    
    df_models = get_model_runs_df()
    
    if len(df_models) > 1:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 🎯 Trade-off: Acurácia Global (NSE) vs Precisão de Incerteza (CRPS)")
            fig_scatter = px.scatter(
                df_models,
                x="crps",
                y="nse",
                size="kge",
                color="model_family",
                hover_name="architecture",
                text="architecture",
                labels={"crps": "CRPS (Menor é Melhor)", "nse": "NSE (Maior é Melhor)"},
                template="plotly_dark"
            )
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, width="stretch")
            
        with c2:
            st.markdown("#### 📉 Comparativo de Erro Absoluto (RMSE vs MAE)")
            fig_bar = px.bar(
                df_models,
                x="architecture",
                y=["rmse", "mae"],
                barmode="group",
                labels={"value": "Erro (m³/s)", "variable": "Métrica"},
                template="plotly_dark",
                color_discrete_sequence=["#ef4444", "#f59e0b"]
            )
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-25)
            st.plotly_chart(fig_bar, width="stretch")
            
    st.markdown("---")
    st.markdown("### 🌊 Simulação de Hidrograma Preditivo com Banda de Incerteza (P10 - P50 - P90)")
    
    np.random.seed(42)
    days = pd.date_range(start="2026-08-01", periods=60, freq="D")
    base_flow = 120 + np.sin(np.linspace(0, 3*np.pi, 60)) * 60 + np.random.normal(0, 8, 60)
    base_flow[35:45] += np.array([20, 65, 140, 220, 180, 110, 60, 30, 15, 5])
    
    observed = base_flow
    pred_p50 = observed + np.random.normal(0, 6, 60)
    pred_p10 = pred_p50 - np.random.uniform(15, 30, 60)
    pred_p90 = pred_p50 + np.random.uniform(18, 38, 60)
    
    fig_hydro = go.Figure()
    
    fig_hydro.add_trace(go.Scatter(
        x=days, y=pred_p90,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Quantil 90%"
    ))
    fig_hydro.add_trace(go.Scatter(
        x=days, y=pred_p10,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.25)",
        name="Intervalo de Confiança (P10-P90)"
    ))
    
    fig_hydro.add_trace(go.Scatter(
        x=days, y=pred_p50,
        mode="lines",
        line=dict(color="#38bdf8", width=2.5),
        name="Previsão Mediana (TFT Model - P50)"
    ))
    
    fig_hydro.add_trace(go.Scatter(
        x=days, y=observed,
        mode="lines+markers",
        line=dict(color="#ffffff", width=2),
        marker=dict(size=4),
        name="Vazão Observada (Estação Fluviométrica)"
    ))
    
    fig_hydro.update_layout(
        title="Hidrograma de Previsão de Vazão vs Observado — Bacia Hidrográfica de Santa Catarina",
        xaxis_title="Data de Observação",
        yaxis_title="Vazão Líquida (m³/s)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_hydro, width="stretch")

# ==========================================
# 9. MODULE 5: CENTRO DE OPERAÇÕES & DB
# ==========================================
elif navigation == "⚙️ Centro de Operações & DB":
    st.title("⚙️ Operações de Banco de Dados & Exportação")
    st.caption("Auditoria, integridade relacional SQLite e snapshots institucionais")
    
    st.markdown("---")
    
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.markdown("### 📦 Exportação de Datasets e Metadados")
        st.markdown("Exporte os snapshots completos do estado do mestrado para documentação e relatórios institucionais.")
        
        conn = get_db_connection()
        
        df_etapas_exp = pd.read_sql("SELECT * FROM etapas", conn)
        csv_etapas = df_etapas_exp.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cronograma (CSV)", data=csv_etapas, file_name="cronograma_mestrado_ufsc.csv", mime="text/csv")
        
        df_models_exp = pd.read_sql("SELECT * FROM model_runs", conn)
        csv_models = df_models_exp.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Model Registry (CSV)", data=csv_models, file_name="model_registry_benchmarks.csv", mime="text/csv")
        
        df_chk_exp = pd.read_sql("SELECT * FROM checklist", conn)
        csv_chk = df_chk_exp.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Checklist Operacional (CSV)", data=csv_chk, file_name="checklist_atividades_ufsc.csv", mime="text/csv")

    with c_right:
        st.markdown("### 🗄️ Query Console SQLite Integrado")
        st.caption("Execução direta de queries analíticas na base do tracker.")
        
        custom_query = st.text_area(
            "SQL Query:",
            value="SELECT model_family, COUNT(*) as total_runs, ROUND(AVG(nse), 3) as avg_nse, ROUND(MIN(rmse), 2) as min_rmse FROM model_runs GROUP BY model_family ORDER BY avg_nse DESC"
        )
        
        if st.button("⚡ Executar SQL Query"):
            try:
                conn = get_db_connection()
                query_result = pd.read_sql(custom_query, conn)
                st.dataframe(query_result, width="stretch")
            except Exception as e:
                st.error(f"Erro na execução da Query: {str(e)}")

    st.markdown("---")
    st.caption("HydraOps v2.4 Enterprise Edition | Arquitetura Desenvolvida para Pós-Graduação PPGCC/UFSC")