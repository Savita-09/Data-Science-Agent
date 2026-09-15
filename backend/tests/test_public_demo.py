import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


@pytest.mark.parametrize('existing_key', ['', 'previous-private-application-key-123456789'])
def test_public_demo_accepts_remote_requests_without_an_api_key(tmp_path, existing_key):
    settings = Settings(
        app_env='production', api_key=existing_key, public_demo=True, ephemeral_storage=True,
        data_root=tmp_path/'data', project_root=tmp_path/'projects',
        max_queued_jobs=1, max_job_seconds=60,
    )
    with TestClient(create_app(settings), client=('203.0.113.5', 50000)) as client:
        health = client.get('/api/health').json()
        assert health['auth_required'] is False
        assert health['public_demo'] is True
        assert client.get('/api/datasets').json() == []
        dataset = client.post('/api/datasets/sample')
        assert dataset.status_code == 201
        payload = {'dataset_id': dataset.json()['id'], 'goal': 'Predict churn',
                   'target': 'churn', 'time_budget_seconds': 60}
        started = client.post('/api/analyses', json=payload)
        assert started.status_code == 202
        assert client.get('/api/analyses/'+started.json()['id']).status_code == 200
        assert client.post('/api/analyses', json=payload).status_code == 429
        assert client.post('/api/analyses', json={**payload, 'time_budget_seconds': 900}).status_code == 422
        assert client.get('/openapi.json').json()['paths']['/api/analyses']['get'].get('security') is None


def test_public_demo_uses_separate_data_and_model_directories(tmp_path):
    roots = {'data_root': tmp_path/'data', 'project_root': tmp_path/'projects'}
    private = Settings(api_key='private-key', **roots)
    with TestClient(create_app(private), headers={'X-API-Key': 'private-key'}) as client:
        private_dataset = client.post('/api/datasets/sample').json()['id']
    public = Settings(public_demo=True, ephemeral_storage=True, **roots)
    assert public.data_root == roots['data_root']/'public-demo'
    assert public.project_root == roots['project_root']/'public-demo'
    # Workers reconstruct Settings from a serialized copy without nesting another directory.
    reconstructed = Settings(**public.model_dump())
    assert reconstructed.data_root == public.data_root
    assert reconstructed.project_root == public.project_root
    with TestClient(create_app(public)) as client:
        assert client.get('/api/datasets').json() == []
        assert client.get('/api/datasets/'+private_dataset).status_code == 404
        assert client.get('/api/analyses').json() == []
    assert (private.data_root/(private_dataset+'.csv')).is_file()


def test_private_production_still_requires_authentication(tmp_path):
    with pytest.raises(ValueError, match='APP_API_KEY'):
        Settings(app_env='production', data_root=tmp_path/'data', project_root=tmp_path/'projects').initialize()
    settings = Settings(app_env='production', api_key='private-application-key-1234567890',
                        data_root=tmp_path/'data', project_root=tmp_path/'projects')
    with TestClient(create_app(settings)) as client:
        assert client.get('/api/analyses').status_code == 401
        assert client.post('/api/datasets/sample').status_code == 401
        assert client.get('/api/health').json()['auth_required'] is True


def test_public_demo_must_explicitly_use_disposable_storage():
    with pytest.raises(ValueError, match='ephemeral_storage'):
        Settings(public_demo=True)
