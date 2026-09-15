from pathlib import Path
import os
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')

class Settings(BaseModel):
    app_env: str = 'development'
    api_key: str = ''
    data_root: Path = ROOT / 'data'
    project_root: Path = ROOT / 'projects'
    max_upload_mb: int = Field(20, ge=1, le=100)
    max_rows: int = Field(50000, ge=50, le=200000)
    max_columns: int = Field(100, ge=2, le=500)
    max_features: int = Field(40, ge=1, le=100)
    max_encoded_features: int = Field(512, ge=10, le=2000)
    max_job_seconds: int = Field(900, ge=10, le=7200)
    max_memory_mb: int = Field(2048, ge=256, le=16384)
    max_queued_jobs: int = Field(20, ge=1, le=100)
    cv_folds: int = Field(3, ge=2, le=5)
    tuning_iterations: int = Field(3, ge=1, le=10)
    shap_rows: int = Field(10, ge=1, le=30)
    seed: int = 42
    cors_origins: list[str] = ['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:5174']
    llm_base_url: str = ''
    llm_api_key: str = ''
    llm_model: str = ''
    llm_timeout_seconds: int = Field(30, ge=1, le=120)
    groq_api_key: str = Field('', repr=False)
    groq_model: str = 'openai/gpt-oss-20b'
    groq_timeout_seconds: int = Field(30, ge=1, le=120)

    @property
    def groq_configured(self): return bool(self.groq_api_key.strip() and self.groq_model.strip())

    @property
    def db_path(self): return self.data_root / 'workspace.sqlite3'

    @property
    def llm_configured(self): return bool(self.llm_base_url and self.llm_api_key and self.llm_model)

    def initialize(self):
        if self.app_env == 'production' and len(self.api_key) < 32:
            raise ValueError('Production requires APP_API_KEY with at least 32 characters.')
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.project_root.mkdir(parents=True, exist_ok=True)

def get_settings():
    path = Path(os.environ.get('APP_CONFIG', ROOT / 'config.yaml'))
    values = yaml.safe_load(path.read_text()) if path.exists() else {}
    for name, env in {'app_env':'APP_ENV', 'api_key':'APP_API_KEY', 'data_root':'DATA_ROOT', 'project_root':'PROJECT_ROOT', 'max_job_seconds':'MAX_JOB_SECONDS', 'max_memory_mb':'MAX_MEMORY_MB', 'llm_base_url':'LLM_BASE_URL', 'llm_api_key':'LLM_API_KEY', 'llm_model':'LLM_MODEL', 'llm_timeout_seconds':'LLM_TIMEOUT_SECONDS', 'groq_api_key':'GROQ_API_KEY', 'groq_model':'GROQ_MODEL', 'groq_timeout_seconds':'GROQ_TIMEOUT_SECONDS'}.items():
        if os.getenv(env): values[name] = os.environ[env]
    if os.getenv('CORS_ORIGINS'): values['cors_origins'] = os.environ['CORS_ORIGINS'].split(',')
    for name in ('data_root', 'project_root'):
        if name in values:
            path = Path(values[name]); values[name] = path if path.is_absolute() else ROOT / path
    return Settings(**values)
