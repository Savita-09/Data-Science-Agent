import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Intelligent Data Science Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:        #0a0e17;
    --surface:   #111827;
    --surface2:  #1c2333;
    --border:    #1f2d45;
    --accent:    #00e5ff;
    --accent2:   #7c3aed;
    --accent3:   #f59e0b;
    --success:   #10b981;
    --danger:    #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'Space Mono', monospace;
    --sans:      'Syne', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.hero {
    background: linear-gradient(135deg, #0d1b2e 0%, #0a0e17 50%, #150d26 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(0,229,255,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 20% 50%, rgba(124,58,237,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-tag {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero h1 {
    font-family: var(--sans) !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    color: #fff !important;
    margin: 0 0 0.5rem !important;
}
.hero h1 span { color: var(--accent); }
.hero p {
    color: var(--muted);
    font-size: 0.82rem;
    max-width: 560px;
    margin: 0;
    font-family: var(--mono);
}

.section-heading {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-heading::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent { border-left: 3px solid var(--accent); }
.card-purple { border-left: 3px solid var(--accent2); }
.card-amber  { border-left: 3px solid var(--accent3); }
.card-green  { border-left: 3px solid var(--success); }

.stat-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1rem; }
.stat-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-family: var(--mono);
    font-size: 0.75rem;
}
.stat-pill .label { color: var(--muted); display: block; font-size: 0.65rem; }
.stat-pill .value { color: var(--accent); font-weight: 700; font-size: 0.9rem; }

.pipeline {
    display: flex;
    gap: 0;
    margin: 1rem 0;
    overflow-x: auto;
}
.pipeline-step {
    flex: 1;
    min-width: 130px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-right: none;
    padding: 0.85rem 1rem;
    position: relative;
    transition: all 0.3s;
}
.pipeline-step:first-child { border-radius: 10px 0 0 10px; }
.pipeline-step:last-child  { border-right: 1px solid var(--border); border-radius: 0 10px 10px 0; }
.pipeline-step.active  { background: rgba(0,229,255,0.08); border-color: var(--accent); }
.pipeline-step.done    { background: rgba(16,185,129,0.08); border-color: var(--success); }
.pipeline-step.pending { opacity: 0.45; }
.step-num  { font-family: var(--mono); font-size: 0.62rem; color: var(--muted); display: block; }
.step-name { font-size: 0.78rem; font-weight: 600; color: var(--text); display: block; margin-top: 0.2rem; }
.step-icon { font-size: 1.1rem; display: block; margin-bottom: 0.2rem; }

.agent-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--muted);
    margin: 0.2rem;
}
.agent-badge.active { color: var(--accent); border-color: var(--accent); }

.output-block {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
    font-family: var(--mono);
    font-size: 0.8rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
}

.terminal {
    background: #050810;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: #4ade80;
    max-height: 280px;
    overflow-y: auto;
    line-height: 1.6;
}

div[data-testid="stFileUploadDropzone"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    color: var(--muted) !important;
}
div[data-testid="stFileUploadDropzone"]:hover { border-color: var(--accent) !important; }

.stButton>button {
    background: linear-gradient(135deg, var(--accent) 0%, #0891b2 100%) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(0,229,255,0.3) !important;
}

.stSelectbox>div>div, .stTextArea>div>div, .stTextInput>div>div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSelectbox label, .stTextArea label, .stTextInput label,
.stCheckbox label, .stRadio label {
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em !important;
}

div[data-testid="stDataFrame"] { background: var(--surface) !important; }
div[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
}
div[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.72rem !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--mono) !important; }

.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.72rem !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

.stExpander { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
.stExpander summary { color: var(--text) !important; font-family: var(--mono) !important; font-size: 0.78rem !important; }

.stProgress > div > div { background: linear-gradient(90deg, var(--accent2), var(--accent)) !important; border-radius: 4px !important; }

hr { border-color: var(--border) !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

.insight-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
}
.tag { font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase; padding: 0.15rem 0.5rem; border-radius: 4px; margin-right: 0.5rem; }
.tag-info    { background: rgba(0,229,255,0.12); color: var(--accent); }
.tag-warn    { background: rgba(245,158,11,0.12); color: var(--accent3); }
.tag-success { background: rgba(16,185,129,0.12); color: var(--success); }
.tag-model   { background: rgba(124,58,237,0.12); color: var(--accent2); }
</style>
""", unsafe_allow_html=True)



PROBLEM_TYPES = {
    "classification": {
        "description": "Predicting categorical outcomes (churn, fraud, diagnosis)",
        "algorithms": ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "Neural Network"],
        "metrics": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        "icon": "🎯"
    },
    "regression": {
        "description": "Predicting continuous values (prices, sales, demand)",
        "algorithms": ["Linear Regression", "Ridge", "Lasso", "Random Forest", "XGBoost"],
        "metrics": ["MAE", "MSE", "RMSE", "R² Score", "MAPE"],
        "icon": "📈"
    },
    "clustering": {
        "description": "Grouping similar data points (segmentation, clustering)",
        "algorithms": ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"],
        "metrics": ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz"],
        "icon": "🔵"
    },
    "time_series": {
        "description": "Temporal predictions (forecasting, trends)",
        "algorithms": ["ARIMA", "Prophet", "LSTM", "XGBoost Time Series"],
        "metrics": ["MAE", "RMSE", "MAPE", "SMAPE"],
        "icon": "⏱️"
    },
    "nlp": {
        "description": "Text analysis (sentiment, classification, NER)",
        "algorithms": ["TF-IDF + Classifier", "BERT", "RoBERTa"],
        "metrics": ["Accuracy", "F1-Score", "BLEU Score"],
        "icon": "📝"
    },
    "anomaly_detection": {
        "description": "Identifying unusual patterns (fraud, monitoring)",
        "algorithms": ["Isolation Forest", "One-Class SVM", "Autoencoder"],
        "metrics": ["Precision", "Recall", "F1-Score", "AUC-ROC"],
        "icon": "🚨"
    },
}

# Valid Groq models (updated list)
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "LLaMA 3.3 70B (Recommended)",
    "llama-3.1-8b-instant": "LLaMA 3.1 8B (Fast)",
    "openai/gpt-oss-120b": "GPT Oss Model"
}

WORKFLOW_STEPS = [
    {"id": "ingest",     "name": "Data Ingestion",    "icon": "📥", "agent": "Data Engineer"},
    {"id": "eda",        "name": "EDA",               "icon": "🔍", "agent": "Data Analyst"},
    {"id": "preprocess", "name": "Preprocessing",     "icon": "⚙️",  "agent": "Data Engineer"},
    {"id": "feature",    "name": "Feature Eng.",      "icon": "🔧", "agent": "ML Engineer"},
    {"id": "train",      "name": "Model Training",    "icon": "🤖", "agent": "ML Engineer"},
    {"id": "evaluate",   "name": "Evaluation",        "icon": "📊", "agent": "ML Engineer"},
    {"id": "insights",   "name": "Business Insights", "icon": "💡", "agent": "Business Analyst"},
    {"id": "deploy",     "name": "Deployment Plan",   "icon": "🚀", "agent": "MLOps Engineer"},
]

AGENTS = ["Data Engineer", "Data Analyst", "ML Engineer", "Business Analyst", "MLOps Engineer"]



def analyze_dataset(df: pd.DataFrame) -> dict:
    info = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": list(df.select_dtypes(include=np.number).columns),
        "categorical_cols": list(df.select_dtypes(include=["object", "category"]).columns),
        "datetime_cols": list(df.select_dtypes(include=["datetime64"]).columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }
    info["num_count"] = len(info["numeric_cols"])
    info["cat_count"] = len(info["categorical_cols"])
    return info


def auto_detect_problem(df: pd.DataFrame, target_col: str) -> str:
    if target_col not in df.columns:
        return "classification"
    col = df[target_col].dropna()
    nunique = col.nunique()
    dtype = col.dtype

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "time_series"
    if pd.api.types.is_numeric_dtype(dtype):
        if nunique <= 10:
            return "classification"
        return "regression"
    if nunique <= 20:
        return "classification"
    return "nlp"


def get_column_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        sample_val = str(s.dropna().iloc[0]) if s.dropna().shape[0] > 0 else "—"
        rows.append({
            "Column": col,
            "Dtype": str(s.dtype),
            "Non-Null": int(s.count()),
            "Null %": f"{s.isnull().mean()*100:.1f}%",
            "Unique": int(s.nunique()),
            "Sample": sample_val[:40],
        })
    return pd.DataFrame(rows)


def render_pipeline(steps_placeholder, current_idx: int, include_deployment: bool = True):
    """Render the pipeline progress bar. FIX: correct step numbering."""
    html = '<div class="pipeline">'
    visible_steps = [s for s in WORKFLOW_STEPS if include_deployment or s["id"] != "deploy"]
    for i, step in enumerate(visible_steps):
        num = str(i + 1).zfill(2)   # FIX: zero-pad correctly (01, 02 ... 10, 11)
        if i < current_idx:
            css, icon = "done", "✅"
        elif i == current_idx:
            css, icon = "active", step["icon"]
        else:
            css, icon = "pending", step["icon"]
        html += f"""
        <div class="pipeline-step {css}">
            <span class="step-icon">{icon}</span>
            <span class="step-num">{num}</span>
            <span class="step-name">{step['name']}</span>
        </div>"""
    html += "</div>"
    steps_placeholder.markdown(html, unsafe_allow_html=True)


def check_groq_key(api_key: str) -> tuple[bool, str]:
    """Validate Groq API key format."""
    if not api_key:
        return False, "No API key provided"
    if not api_key.startswith("gsk_"):
        return False, "Groq API keys start with 'gsk_'"
    if len(api_key) < 40:
        return False, "API key appears too short"
    return True, "OK"



def run_crewai_workflow(
    problem_type: str,
    dataset_description: str,
    df_info: dict,
    custom_requirements: str,
    include_deployment: bool,
    groq_api_key: str,
    groq_model: str,
    steps_placeholder,
    progress_placeholder,
    log_placeholder,
) -> str:
    """
    Run the CrewAI multi-agent workflow with Groq LLM.
    FIX: Removed threading (caused Streamlit state issues). Runs synchronously.
    FIX: Correct Groq model names and base URL.
    FIX: Proper error messages.
    """
    try:
        from crewai import Agent, Task, Crew, Process, LLM
    except ImportError:
        return (
            "## ❌ CrewAI Not Installed\n\n"
            "Install the required packages:\n\n"
            "```bash\n"
            "pip install crewai groq\n"
            "```\n\n"
            "Then restart the Streamlit app."
        )

    logs = []

    def log(msg: str):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        log_text = "\n".join(logs[-30:])
        log_placeholder.markdown(
            f'<div class="terminal">{log_text}</div>',
            unsafe_allow_html=True
        )

    try:
        log("🔌 Initialising Groq LLM via CrewAI...")
        render_pipeline(steps_placeholder, 0, include_deployment)
        progress_placeholder.progress(5, text="Connecting to Groq LLM…")

        # FIX: Correct Groq LLM configuration
        # CrewAI uses LiteLLM under the hood; prefix with "groq/" for Groq models
        llm = LLM(
            model=f"groq/{groq_model}",
            api_key=groq_api_key,
            temperature=0.1,
            max_tokens=4096,
        )

        problem_info = PROBLEM_TYPES.get(problem_type, PROBLEM_TYPES["classification"])
        context_block = f"""
Dataset statistics:
- Rows: {df_info['rows']:,}  |  Columns: {df_info['cols']}
- Numeric features: {df_info['num_count']}  |  Categorical: {df_info['cat_count']}
- Missing values: {df_info['missing_pct']}%
- Duplicates: {df_info['duplicates']}
- Memory: {df_info['memory_mb']} MB

Numeric columns  : {', '.join(df_info['numeric_cols'][:12]) or 'None'}
Categorical cols : {', '.join(df_info['categorical_cols'][:8]) or 'None'}

Business context:
{dataset_description}
"""

        # Agents
        log("🤖 Spawning specialist agents…")
        progress_placeholder.progress(12, text="Spawning agents…")

        data_engineer = Agent(
            role="Senior Data Engineer",
            goal=(
                f"Prepare and validate the dataset for {problem_type} modeling. "
                "Perform data quality checks, missing-value imputation, outlier treatment, "
                "feature encoding, scaling, and train/test splitting. "
                "Document every transformation clearly."
            ),
            backstory="Expert data engineer with 10+ years specialising in production-ready ML pipelines.",
            verbose=True,
            llm=llm,
            max_iter=3,         # limit iterations to avoid infinite loops / token waste
        )

        data_analyst = Agent(
            role="Data Analyst",
            goal=(
                f"Perform exploratory data analysis for {problem_type}. "
                "Provide descriptive statistics, distribution insights, correlation findings, "
                "class-balance information, and key patterns."
            ),
            backstory="Expert analyst with 8+ years. Finds the story hidden in data.",
            verbose=True,
            llm=llm,
            max_iter=3,
        )

        ml_engineer = Agent(
            role="Machine Learning Engineer",
            goal=(
                f"Build and evaluate models for {problem_type}. "
                f"Train algorithms: {', '.join(problem_info['algorithms'])}. "
                f"Report metrics: {', '.join(problem_info['metrics'])}. "
                "Use 5-fold cross-validation, hyperparameter tuning, feature importance."
            ),
            backstory="PhD-level ML engineer balancing complexity, interpretability, and performance.",
            verbose=True,
            llm=llm,
            max_iter=3,
        )

        business_analyst = Agent(
            role="Business Analyst",
            goal=(
                "Translate ML findings into actionable business recommendations. "
                "Provide executive summary, ROI analysis, KPI framework, and implementation roadmap."
            ),
            backstory="10+ years bridging data science and C-suite. Clear, concise communicator.",
            verbose=True,
            llm=llm,
            max_iter=3,
        )

        mlops_engineer = Agent(
            role="MLOps Engineer",
            goal=(
                "Design a production deployment plan: model serialisation, REST API spec, "
                "CI/CD pipeline, drift monitoring, and documentation."
            ),
            backstory="8+ years in MLOps. Expert in Docker, MLflow, and cloud platforms.",
            verbose=True,
            llm=llm,
            max_iter=3,
        )

        # Tasks 
        log("📋 Defining workflow tasks…")
        render_pipeline(steps_placeholder, 1, include_deployment)
        progress_placeholder.progress(20, text="Defining tasks…")

        task_prepare = Task(
            description=f"""Prepare the dataset for {problem_type} modelling.

{context_block}

Provide:
1. Data quality assessment (nulls, types, outliers, duplicates)
2. Imputation strategy per column type
3. Outlier detection (IQR / z-score)
4. Encoding plan for categorical variables
5. Feature scaling recommendations
6. Feature engineering suggestions
7. Train / Validation / Test split strategy (e.g. 70/15/15)
8. A concise scikit-learn Pipeline code snippet
9. Data readiness score (0-100)

Additional requirements: {custom_requirements or 'None'}""",
            expected_output=(
                "Structured data preparation report with: Dataset Quality Score, "
                "missing-value strategy, outlier summary, feature engineering log, "
                "split configuration, and a scikit-learn Pipeline code snippet."
            ),
            agent=data_engineer,
        )

        task_eda = Task(
            description=f"""Perform EDA on the dataset for {problem_type}.

{context_block}

Analyse:
1. Descriptive statistics (mean, median, std, skew, kurtosis)
2. Distribution shapes per feature
3. Correlation matrix highlights (top 10 pairs)
4. Target variable analysis (class balance / distribution)
5. Key patterns and anomalies
6. Recommended visualisations with rationale""",
            expected_output=(
                "EDA report with: summary statistics table, top correlated features, "
                "target distribution insights, 5+ key findings with business interpretation, "
                "and recommended chart types."
            ),
            agent=data_analyst,
            context=[task_prepare],
        )

        task_model = Task(
            description=f"""Build and evaluate ML models for {problem_type}.

{context_block}

Steps:
1. Baseline model (DummyClassifier/Regressor)
2. Train: {', '.join(problem_info['algorithms'])}
3. 5-fold cross-validation per model
4. Metrics: {', '.join(problem_info['metrics'])}
5. Hyperparameter tuning recommendations
6. Top-10 feature importances
7. Final model selection with justification and code snippet

Additional requirements: {custom_requirements or 'None'}""",
            expected_output=(
                f"Model evaluation report with: comparison table (all models × metrics), "
                "best model justification, hyperparameter recommendations, "
                "feature importance ranking, and code snippet."
            ),
            agent=ml_engineer,
            context=[task_prepare, task_eda],
        )

        task_business = Task(
            description=f"""Write an executive business report for the {problem_type} analysis.

Steps:
1. Executive summary (3 concise bullets)
2. Problem → solution narrative
3. Business impact quantification (revenue / cost / risk)
4. Prioritised actionable recommendations
5. KPI framework
6. 90-day implementation roadmap
7. Risks and mitigations

Additional requirements: {custom_requirements or 'None'}""",
            expected_output=(
                "Executive business report in Markdown with sections: "
                "Executive Summary, Business Impact, Recommendations, "
                "KPI Framework, 90-Day Roadmap, Risk Register."
            ),
            agent=business_analyst,
            context=[task_model],
        )

        all_agents = [data_engineer, data_analyst, ml_engineer, business_analyst]
        all_tasks  = [task_prepare, task_eda, task_model, task_business]

        if include_deployment:
            task_deploy = Task(
                description=f"""Design a production deployment plan for the {problem_type} model.

Steps:
1. Model serialisation (joblib / ONNX / TorchScript)
2. REST API design (FastAPI endpoints with payload schemas)
3. Docker containerisation spec
4. CI/CD pipeline (GitHub Actions YAML snippet)
5. Drift detection strategy (data + concept drift)
6. Monitoring dashboard KPIs
7. Documentation template and rollback strategy""",
                expected_output=(
                    "Deployment blueprint with: API specification, Dockerfile template, "
                    "CI/CD YAML snippet, monitoring plan, and rollback strategy."
                ),
                agent=mlops_engineer,
                context=[task_model],
            )
            all_agents.append(mlops_engineer)
            all_tasks.append(task_deploy)

        # Crew 
        crew = Crew(
            agents=all_agents,
            tasks=all_tasks,
            process=Process.sequential,
            verbose=True,
        )

        task_count = len(all_tasks)
        progress_per_task = int(70 / task_count)
        task_labels = [
            "Data Engineer preparing dataset…",
            "Data Analyst running EDA…",
            "ML Engineer training models…",
            "Business Analyst writing report…",
            "MLOps Engineer designing deployment…",
        ]

        log("🚀 Crew kickoff — sequential workflow started…")
        render_pipeline(steps_placeholder, 2, include_deployment)
        progress_placeholder.progress(25, text="Crew is working…")

        # CrewAI's sequential process is already sequential — no need for threads.
        # Threading with Streamlit causes session state issues and broken UI updates.
        for i in range(task_count):
            label = task_labels[i] if i < len(task_labels) else "Processing…"
            log(f"  ⏳ {label}")
            render_pipeline(steps_placeholder, i + 2, include_deployment)
            progress_placeholder.progress(25 + i * progress_per_task, text=label)

        log("⚡ Running crew.kickoff()…")
        result = crew.kickoff()

        render_pipeline(steps_placeholder, len(WORKFLOW_STEPS), include_deployment)
        progress_placeholder.progress(100, text="✅ Workflow complete!")
        log("✅ All tasks completed successfully.")

        return str(result)

    except Exception as e:
        err_msg = str(e)
        log(f"❌ Error: {err_msg}")

        if "401" in err_msg or "authentication" in err_msg.lower() or "api_key" in err_msg.lower():
            return (
                "## ❌ Authentication Error\n\n"
                "Your Groq API key is invalid or expired.\n\n"
                "1. Go to [console.groq.com](https://console.groq.com)\n"
                "2. Generate a new API key\n"
                "3. Paste it in the sidebar (starts with `gsk_`)\n\n"
                f"**Raw error:** `{err_msg}`"
            )
        elif "rate_limit" in err_msg.lower() or "429" in err_msg:
            return (
                "## ❌ Rate Limit Exceeded\n\n"
                "Groq free tier has rate limits. Try:\n"
                "- Switching to `llama-3.1-8b-instant` (higher rate limits)\n"
                "- Waiting 60 seconds and retrying\n"
                "- Upgrading your Groq plan\n\n"
                f"**Raw error:** `{err_msg}`"
            )
        elif "model" in err_msg.lower() and ("not found" in err_msg.lower() or "does not exist" in err_msg.lower()):
            return (
                "## ❌ Model Not Found\n\n"
                f"The selected model is unavailable on Groq. "
                "Please select a different model in the sidebar.\n\n"
                f"**Raw error:** `{err_msg}`"
            )
        elif "litellm" in err_msg.lower() or "fallback" in err_msg.lower() or "native provider" in err_msg.lower():
            return (
                "## ❌ LiteLLM Required\n\n"
                "CrewAI needs the `litellm` package to connect to Groq.\n\n"
                "```bash\npip install litellm\n```\n\n"
                "Then restart the app. Falling back to direct Groq API…\n\n"
                f"**Raw error:** `{err_msg}`"
            )
        else:
            return f"## ❌ Workflow Error\n\n```\n{err_msg}\n```"



def run_groq_direct(
    problem_type: str,
    dataset_description: str,
    df_info: dict,
    custom_requirements: str,
    include_deployment: bool,
    groq_api_key: str,
    groq_model: str,
    steps_placeholder,
    progress_placeholder,
    log_placeholder,
) -> str:
    """
    Direct Groq API fallback when CrewAI is not installed.
    Simulates the multi-agent pipeline with sequential prompts.
    """
    try:
        from groq import Groq
    except ImportError:
        return (
            "## ❌ Neither CrewAI nor Groq SDK is installed\n\n"
            "Install at least one:\n\n"
            "```bash\n"
            "pip install groq           # lightweight, direct API\n"
            "pip install crewai groq    # full multi-agent\n"
            "```"
        )

    client = Groq(api_key=groq_api_key)
    problem_info = PROBLEM_TYPES.get(problem_type, PROBLEM_TYPES["classification"])
    logs = []

    def log(msg: str):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        log_placeholder.markdown(
            f'<div class="terminal">{"<br>".join(logs[-30:])}</div>',
            unsafe_allow_html=True
        )

    def call_groq(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        return resp.choices[0].message.content

    context = (
        f"Dataset: {df_info['rows']:,} rows × {df_info['cols']} columns | "
        f"Numeric: {df_info['num_count']} | Categorical: {df_info['cat_count']} | "
        f"Missing: {df_info['missing_pct']}% | Duplicates: {df_info['duplicates']} | "
        f"Memory: {df_info['memory_mb']} MB\n"
        f"Numeric cols: {', '.join(df_info['numeric_cols'][:10]) or 'None'}\n"
        f"Categorical cols: {', '.join(df_info['categorical_cols'][:6]) or 'None'}\n"
        f"Business context: {dataset_description}\n"
        f"Additional requirements: {custom_requirements or 'None'}"
    )

    report_parts = []
    agent_steps = [
        (
            "Data Engineer",
            "You are a Senior Data Engineer specialising in ML pipelines.",
            f"Prepare a data preprocessing report for {problem_type} modelling.\n\n{context}\n\n"
            f"Cover: quality assessment, imputation strategy, outlier handling, encoding, "
            f"scaling, feature engineering, train/test split, and a scikit-learn Pipeline snippet. "
            f"Format as clean Markdown with a ## Data Preparation heading.",
            1,
            "Data Engineer preparing dataset…",
        ),
        (
            "Data Analyst",
            "You are an expert Data Analyst.",
            f"Perform EDA for {problem_type}.\n\n{context}\n\n"
            f"Cover: descriptive statistics, distributions, correlations (top 10), "
            f"target analysis, patterns, anomalies, and recommended visualisations. "
            f"Format as clean Markdown with a ## Exploratory Data Analysis heading.",
            2,
            "Data Analyst running EDA…",
        ),
        (
            "ML Engineer",
            "You are a PhD-level Machine Learning Engineer.",
            f"Write a model evaluation report for {problem_type}.\n\n{context}\n\n"
            f"Algorithms: {', '.join(problem_info['algorithms'])}.\n"
            f"Metrics: {', '.join(problem_info['metrics'])}.\n"
            f"Cover: baseline, model comparison table, 5-fold CV, hyperparameter tuning, "
            f"feature importance, best model justification, and code snippet. "
            f"Format as clean Markdown with a ## Model Evaluation heading.",
            3,
            "ML Engineer training models…",
        ),
        (
            "Business Analyst",
            "You are a senior Business Analyst bridging data science and strategy.",
            f"Write an executive business report for the {problem_type} analysis.\n\n{context}\n\n"
            f"Cover: executive summary, business impact, recommendations, KPI framework, "
            f"90-day roadmap, and risk register. "
            f"Format as clean Markdown with a ## Business Insights heading.",
            4,
            "Business Analyst writing report…",
        ),
    ]

    if include_deployment:
        agent_steps.append((
            "MLOps Engineer",
            "You are an MLOps Engineer specialising in production ML systems.",
            f"Write a deployment plan for the {problem_type} model.\n\n{context}\n\n"
            f"Cover: model serialisation, FastAPI spec, Docker spec, CI/CD YAML, "
            f"drift detection, monitoring KPIs, and rollback strategy. "
            f"Format as clean Markdown with a ## Deployment Plan heading.",
            5,
            "MLOps Engineer designing deployment…",
        ))

    total = len(agent_steps)
    for agent_name, sys_prompt, user_prompt, step_num, step_label in agent_steps:
        log(f"🤖 [{agent_name}] {step_label}")
        render_pipeline(steps_placeholder, step_num, include_deployment)
        pct = int(20 + (step_num / total) * 70)
        progress_placeholder.progress(pct, text=step_label)

        try:
            part = call_groq(sys_prompt, user_prompt)
            report_parts.append(part)
            log(f"  ✅ [{agent_name}] Done ({len(part)} chars)")
        except Exception as e:
            log(f"  ❌ [{agent_name}] Error: {e}")
            report_parts.append(f"## {agent_name} Error\n\n`{e}`")

    render_pipeline(steps_placeholder, len(WORKFLOW_STEPS), include_deployment)
    progress_placeholder.progress(100, text="✅ Workflow complete!")
    log("✅ All agents completed.")

    header = (
        f"# AutoDS Analysis Report\n\n"
        f"**Problem Type:** {problem_type.replace('_', ' ').title()}  \n"
        f"**Model:** {groq_model}  \n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"**Dataset:** {df_info['rows']:,} rows × {df_info['cols']} columns\n\n"
        f"---\n\n"
    )
    return header + "\n\n---\n\n".join(report_parts)


def run_workflow(
    problem_type, dataset_description, df_info, custom_requirements,
    include_deployment, groq_api_key, groq_model,
    steps_placeholder, progress_placeholder, log_placeholder
) -> str:
    """Try CrewAI first; fall back to direct Groq SDK on any error."""
    try:
        import crewai  # noqa: F401
        result = run_crewai_workflow(
            problem_type, dataset_description, df_info, custom_requirements,
            include_deployment, groq_api_key, groq_model,
            steps_placeholder, progress_placeholder, log_placeholder,
        )
        # If CrewAI returned an error message (starts with ## ❌), fall back
        if isinstance(result, str) and result.startswith("## ❌"):
            raise RuntimeError("CrewAI workflow failed, falling back to direct Groq")
        return result
    except (ImportError, RuntimeError):
        return run_groq_direct(
            problem_type, dataset_description, df_info, custom_requirements,
            include_deployment, groq_api_key, groq_model,
            steps_placeholder, progress_placeholder, log_placeholder,
        )


# Session State
for key, default in {
    "df": None, "df_info": None, "filename": "",
    "result": None, "running": False, "target_col": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


#Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-family:var(--sans);font-size:1.3rem;font-weight:800;color:#fff;">
            Configuration
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-heading">🔑 Groq API Key</div>', unsafe_allow_html=True)
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="gsk_…",
        help="Get your free key at console.groq.com",
    )
    if groq_key:
        valid, reason = check_groq_key(groq_key)
        if valid:
            st.success("✅ groq api uploaded", icon="🔐")
        else:
            st.warning(f"⚠️ {reason}", icon="🔑")

    st.divider()


    st.markdown('<div class="section-heading">🧠 Groq Model</div>', unsafe_allow_html=True)
    groq_model = st.selectbox(
        "Select model",
        options=list(GROQ_MODELS.keys()),
        format_func=lambda x: GROQ_MODELS[x],
        index=0,
    )

    st.divider()

    st.markdown('<div class="section-heading">🎯 Problem Type</div>', unsafe_allow_html=True)
    problem_choice = st.selectbox(
        "Select problem type",
        options=list(PROBLEM_TYPES.keys()),
        format_func=lambda x: f"{PROBLEM_TYPES[x]['icon']}  {x.replace('_', ' ').title()}",
    )
    st.markdown(
        f'<div style="font-family:var(--mono);font-size:0.7rem;color:var(--muted);'
        f'padding:0.4rem 0;">{PROBLEM_TYPES[problem_choice]["description"]}</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown('<div class="section-heading">🎯 Target Column</div>', unsafe_allow_html=True)
    if st.session_state.df is not None:
        cols = list(st.session_state.df.columns)
        target_col = st.selectbox("Select target / label column", ["— none —"] + cols)
        st.session_state.target_col = None if target_col == "— none —" else target_col
    else:
        st.caption("Upload a dataset first")

    st.divider()

    st.markdown('<div class="section-heading">⚙️ Options</div>', unsafe_allow_html=True)
    include_deployment = st.checkbox("Include deployment plan (MLOps)", value=False)
    auto_detect = st.checkbox("Auto-detect problem type", value=True)

    st.divider()

    st.markdown('<div class="section-heading">📌 Custom Requirements</div>', unsafe_allow_html=True)
    custom_req = st.text_area(
        "Additional instructions for agents",
        placeholder="e.g. Focus on feature interactions\nUse SHAP for interpretability\nOptimise for recall",
        height=100,
    )

    st.divider()

    st.markdown('<div class="section-heading">🤖 Active Agents</div>', unsafe_allow_html=True)
    for agent in AGENTS:
        if agent == "MLOps Engineer" and not include_deployment:
            continue
        st.markdown(f'<span class="agent-badge active">● {agent}</span>', unsafe_allow_html=True)

    

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">Multi-Agent AI System</div>
    <h1>Auto<span>DS</span></h1>
    <p>Upload a CSV, Excel, or JSON file and watch specialist AI agents handle the complete data science
    pipeline — from raw data to actionable business insights.</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📥 Dataset Upload</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop your dataset here (CSV, Excel, JSON) or click to browse",
    type=["csv", "xlsx", "xls", "json"],
    label_visibility="collapsed",
)

if uploaded is not None and uploaded.name != st.session_state.filename:
    with st.spinner("Parsing dataset…"):
        try:
            name_lower = uploaded.name.lower()
            if name_lower.endswith(".csv"):
                df = pd.read_csv(uploaded)
            elif name_lower.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded)
            elif name_lower.endswith(".json"):
                df = pd.read_json(uploaded)
            else:
                st.error(f"Unsupported file format: {uploaded.name}")
                st.stop()

            st.session_state.df       = df
            st.session_state.filename = uploaded.name
            st.session_state.df_info  = analyze_dataset(df)
            st.session_state.result   = None
        except Exception as e:
            st.error(f"Failed to parse file: {e}")


# Dataset Preview 
if st.session_state.df is not None:
    df   = st.session_state.df
    info = st.session_state.df_info

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill"><span class="label">Filename</span><span class="value">{st.session_state.filename}</span></div>
        <div class="stat-pill"><span class="label">Rows</span><span class="value">{info['rows']:,}</span></div>
        <div class="stat-pill"><span class="label">Columns</span><span class="value">{info['cols']}</span></div>
        <div class="stat-pill"><span class="label">Numeric</span><span class="value">{info['num_count']}</span></div>
        <div class="stat-pill"><span class="label">Categorical</span><span class="value">{info['cat_count']}</span></div>
        <div class="stat-pill"><span class="label">Missing %</span><span class="value">{info['missing_pct']}%</span></div>
        <div class="stat-pill"><span class="label">Duplicates</span><span class="value">{info['duplicates']}</span></div>
        <div class="stat-pill"><span class="label">Memory</span><span class="value">{info['memory_mb']} MB</span></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋  Data Preview", "📊  Column Stats", "🔬  Quick EDA"])

    with tab1:
        st.dataframe(df.head(50), use_container_width=True, height=320)

    with tab2:
        st.dataframe(get_column_stats(df), use_container_width=True, hide_index=True)

    with tab3:
        num_cols = info["numeric_cols"]
        if num_cols:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Mean (1st numeric)", f"{df[num_cols[0]].mean():.3f}")
            with c2:
                st.metric("Std Dev", f"{df[num_cols[0]].std():.3f}")
            with c3:
                st.metric("Skewness", f"{df[num_cols[0]].skew():.3f}")
            with st.expander("📈 Descriptive Statistics"):
                st.dataframe(df[num_cols].describe().T.round(4), use_container_width=True)
            with st.expander("🔗 Correlation Matrix (numeric)"):
                st.dataframe(df[num_cols].corr().round(3), use_container_width=True)
        else:
            st.info("No numeric columns found for quick EDA.")

    # Auto-detect banner
    if auto_detect and st.session_state.target_col:
        detected = auto_detect_problem(df, st.session_state.target_col)
        if detected != problem_choice:
            st.markdown(
                f'<div class="insight-card">'
                f'<span class="tag tag-info">Auto-Detect</span>'
                f'Detected problem type: <strong style="color:var(--accent)">'
                f'{detected.replace("_", " ").title()}</strong> '
                f'from target column <code>{st.session_state.target_col}</code>. '
                f'Override in sidebar if needed.</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # Pipeline overview
    st.markdown('<div class="section-heading">🔄 Automated Workflow Pipeline</div>', unsafe_allow_html=True)
    steps_ph = st.empty()
    render_pipeline(steps_ph, -1, include_deployment)   # all pending initially

    # Dataset description
    st.markdown('<div class="section-heading">📝 Dataset Description</div>', unsafe_allow_html=True)
    auto_desc = (
        f"Dataset '{st.session_state.filename}' with {info['rows']:,} rows and {info['cols']} columns. "
        f"Numeric: {', '.join(info['numeric_cols'][:8]) or 'None'}. "
        f"Categorical: {', '.join(info['categorical_cols'][:5]) or 'None'}. "
        f"Missing: {info['missing_pct']}%."
    )
    dataset_desc = st.text_area(
        "Describe your dataset and business context",
        value=auto_desc,
        height=90,
        label_visibility="collapsed",
    )

    col_btn1, col_btn2 = st.columns([2, 6])
    with col_btn1:
        run_btn = st.button(
            "⚡  Launch AI Agents",
            use_container_width=True,
            disabled=st.session_state.running,
        )
    with col_btn2:
        if not groq_key:
            st.warning("⚠️  Enter your Groq API key in the sidebar to proceed.")
        else:
            valid, reason = check_groq_key(groq_key)
            if not valid:
                st.error(f"❌ Invalid API key: {reason}")

    # Execution 
    if run_btn and groq_key:
        valid, reason = check_groq_key(groq_key)
        if not valid:
            st.error(f"❌ Cannot launch: {reason}")
        else:
            st.session_state.running = True
            st.session_state.result  = None

            progress_ph = st.empty()
            log_ph      = st.empty()

            progress_ph.progress(0, text="Initialising…")
            log_ph.markdown(
                '<div class="terminal">▶ Starting AutoDS workflow…</div>',
                unsafe_allow_html=True
            )

            final_problem = problem_choice
            if auto_detect and st.session_state.target_col:
                final_problem = auto_detect_problem(df, st.session_state.target_col)

            result = run_workflow(
                problem_type=final_problem,
                dataset_description=dataset_desc,
                df_info=info,
                custom_requirements=custom_req,
                include_deployment=include_deployment,
                groq_api_key=groq_key,
                groq_model=groq_model,
                steps_placeholder=steps_ph,
                progress_placeholder=progress_ph,
                log_placeholder=log_ph,
            )

            st.session_state.result  = result
            st.session_state.running = False
            st.rerun()

    # Results
    if st.session_state.result:
        st.divider()
        st.markdown('<div class="section-heading">📄 Analysis Report</div>', unsafe_allow_html=True)

        result_str = st.session_state.result

        section_markers = {
            "## Data Preparation":       "Data Preparation",
            "## Exploratory Data Analysis": "EDA",
            "## Model Evaluation":       "Model Evaluation",
            "## Business Insights":      "Business Insights",
            "## Deployment Plan":        "Deployment Plan",
        }

        found_sections = {}
        lines = result_str.split("\n")
        current_section = None
        current_lines   = []

        for line in lines:
            matched = next((v for k, v in section_markers.items() if line.strip().startswith(k)), None)
            if matched:
                if current_section and current_lines:
                    found_sections[current_section] = "\n".join(current_lines)
                current_section = matched
                current_lines   = [line]
            elif current_section:
                current_lines.append(line)

        if current_section and current_lines:
            found_sections[current_section] = "\n".join(current_lines)

        if len(found_sections) >= 2:
            tab_labels = ["📑 Full Report"] + [f"  {k}" for k in found_sections]
            result_tabs = st.tabs(tab_labels)
            with result_tabs[0]:
                st.markdown(result_str)
            for i, (sec_name, sec_content) in enumerate(found_sections.items(), start=1):
                with result_tabs[i]:
                    st.markdown(sec_content)
        else:
            st.markdown(result_str)

        # Downloads
        st.divider()
        col_d1, col_d2 = st.columns(2)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with col_d1:
            st.download_button(
                "⬇️  Download Report (.md)",
                data=result_str.encode(),
                file_name=f"autods_report_{ts}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                "⬇️  Download Report (.txt)",
                data=result_str.encode(),
                file_name=f"autods_report_{ts}.txt",
                mime="text/plain",
                use_container_width=True,
            )


else:
    st.markdown("""
    <div class="card card-accent" style="text-align:center;padding:3rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">📂</div>
        <div style="font-family:var(--sans);font-size:1.2rem;font-weight:700;color:#fff;margin-bottom:0.5rem;">
            No Dataset Loaded
        </div>
        <div style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);max-width:400px;margin:0 auto;">
            Upload a dataset file (CSV, Excel, or JSON) above to begin. The AI agents will automatically
            analyse your data and run the full data science workflow.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-heading">✨ What AutoDS Does</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    features = [
        ("📥", "Data Ingestion",  "Auto-parses CSV, Excel, JSON; detects types, assesses quality"),
        ("🔍", "Smart EDA",       "Distributions, correlations, patterns, anomalies"),
        ("🤖", "Model Training",  "Trains multiple algorithms with cross-validation"),
        ("💡", "Insights",        "Translates results into business recommendations"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="card card-accent">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="font-weight:700;margin:0.4rem 0;font-size:0.9rem;">{title}</div>
                <div style="font-family:var(--mono);font-size:0.7rem;color:var(--muted);">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-heading">🚀 Quick Start</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-purple">
        <div style="font-family:var(--mono);font-size:0.78rem;line-height:1.9;">
            <strong style="color:var(--accent);">1.</strong> Get a free API key at
            <a href="https://console.groq.com" target="_blank" style="color:var(--accent2);">console.groq.com</a><br>
            <strong style="color:var(--accent);">2.</strong> Paste it in the sidebar (starts with <code>gsk_</code>)<br>
            <strong style="color:var(--accent);">3.</strong> Upload your dataset (CSV, Excel, or JSON) above<br>
            <strong style="color:var(--accent);">4.</strong> Select your target column &amp; problem type<br>
            <strong style="color:var(--accent);">5.</strong> Click <strong>⚡ Launch AI Agents</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
