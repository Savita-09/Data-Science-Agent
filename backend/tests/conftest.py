import pytest
from fastapi.testclient import TestClient
from app.settings import Settings
from app.main import create_app

@pytest.fixture
def workspace(tmp_path):
    settings=Settings(data_root=tmp_path/'data',project_root=tmp_path/'projects',api_key='test-secret',shap_rows=3,max_job_seconds=120)
    app=create_app(settings)
    with TestClient(app,headers={'X-API-Key':'test-secret'}) as client:yield settings,app.state.store,client
