import pandas as pd
import pytest
from pypdf import PdfReader
from sklearn.datasets import make_regression,make_blobs
from app.agents.orchestrator import OrchestratorAgent
from app.agents.stages import TrainingAgent

def execute(workspace,frame=None,task='classification',target='churn'):
    settings,store,client=workspace
    dataset=client.post('/api/datasets/sample').json() if frame is None else client.post('/api/datasets',files={'file':('synthetic.csv',frame.to_csv(index=False).encode(),'text/csv')}).json()
    response=client.post('/api/analyses',json={'dataset_id':dataset['id'],'goal':'Predict customer churn' if task=='classification' else 'Predict target value' if task=='regression' else 'Segment similar customers','task':task,'target':target,'time_budget_seconds':120,'tuning_iterations':1,'cv_folds':2,'quick':True})
    assert response.status_code==202,response.text
    job=response.json()['id'];assert store.claim()==job
    OrchestratorAgent(settings).run(job)
    record=store.analysis(job)
    assert record['status']=='completed',record['error']
    return job,record['result']

def test_customer_churn_full_workflow_reports_predictions_and_resume(workspace,monkeypatch):
    settings,store,client=workspace;job,result=execute(workspace)
    metrics=result['evaluation']['metrics']
    assert metrics['roc_auc']>.75
    assert metrics['macro_f1']>result['evaluation']['baseline']['macro_f1']
    assert len(store.analysis(job)['checkpoints'])==12
    assert result['explainability']['status']=='completed'
    assert len(result['explainability']['global'])>0
    assert max(row['additivity_error'] for row in result['explainability']['local'])<.01
    inputs={col['name']:col['example'] for col in result['model_schema']}
    prediction=client.post(f'/api/analyses/{job}/predict',json={'rows':[inputs]})
    assert prediction.status_code==200,prediction.text
    assert prediction.json()['predictions'][0] in ('Yes','No')
    assert client.post(f'/api/analyses/{job}/predict',json={'rows':[{'unknown':1}]}).status_code==422
    for kind in ('html','pdf','json','predictions','model'):
        assert client.get(f'/api/analyses/{job}/artifacts/{kind}').status_code==200
    assert len(PdfReader(settings.project_root/job/'report.pdf').pages)>=2
    assert 'Cross-validation leaderboard' in ''.join(page.extract_text() for page in PdfReader(settings.project_root/job/'report.pdf').pages)
    assert client.post(f'/api/analyses/{job}/chat',json={'question':'Which features matter?'}).status_code==200
    store.update(job,status='failed');store.retry(job);store.claim()
    monkeypatch.setattr(TrainingAgent,'run',lambda *args:(_ for _ in ()).throw(AssertionError('Cached training must not rerun')))
    OrchestratorAgent(settings).run(job)
    assert store.analysis(job)['status']=='completed'
    assert all(cp['duration']==0 for cp in store.analysis(job)['checkpoints'])

@pytest.mark.parametrize('task',['regression','clustering'])
def test_other_supported_tasks(workspace,task):
    if task=='regression':
        X,y=make_regression(n_samples=180,n_features=4,noise=8,random_state=7);frame=pd.DataFrame(X,columns=list('abcd'));frame['target']=y
    else:
        X,_=make_blobs(n_samples=180,n_features=2,centers=3,cluster_std=.5,random_state=7);frame=pd.DataFrame(X,columns=['spend','frequency'])
    job,result=execute(workspace,frame,task,'target' if task=='regression' else None)
    assert result['evaluation']['metrics']['r2']>.8 if task=='regression' else result['evaluation']['metrics']['silhouette']>.5
    assert result['explainability']['status']=='completed'
    settings,store,client=workspace
    inputs={col['name']:col['example'] for col in result['model_schema']}
    assert client.post(f'/api/analyses/{job}/predict',json={'rows':[inputs]}).status_code==200
