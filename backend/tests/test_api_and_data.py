import pandas as pd
import pytest
from app.settings import Settings
from app.preprocessing.pipeline import make_preprocessor,choose_features

def test_authentication_and_production_guard(workspace):
    _,_,client=workspace
    assert client.get('/api/health').status_code==200
    assert client.get('/api/analyses',headers={'X-API-Key':'wrong'}).status_code==401
    assert client.get('/api/analyses/../../env').status_code==404
    with pytest.raises(ValueError,match='APP_API_KEY'):Settings(app_env='production').initialize()

def test_upload_validation_and_resource_configuration(workspace):
    settings,store,client=workspace
    assert client.post('/api/datasets',files={'file':('bad.json',b'[{"nested":{"a":1}}]','application/json')}).status_code==422
    assert client.post('/api/datasets',files={'file':('bad.csv',b'x,x\n1,2','text/csv')}).status_code==422
    data=client.post('/api/datasets/sample').json()
    response=client.post('/api/analyses',json={'dataset_id':data['id'],'goal':'Predict churn','time_budget_seconds':121})
    assert response.status_code==422
    response=client.post('/api/analyses',json={'dataset_id':data['id'],'goal':'Predict churn','target':'nonexistent','time_budget_seconds':120})
    assert response.status_code==422
    assert client.post('/api/analyses',json={'dataset_id':data['id'],'goal':'Predict churn','use_llm':True,'time_budget_seconds':120}).status_code==422

def test_fold_preprocessing_and_leakage_guards():
    train=pd.DataFrame({'amount':[1.,2.,3.,None],'region':['A','B','A',None]})
    transformer=make_preprocessor(train);transformer.fit(train)
    before=transformer.named_transformers_['numeric'].named_steps['impute'].statistics_.copy()
    test=pd.DataFrame({'amount':[1e8],'region':['unseen']});assert transformer.transform(test).shape[0]==1
    assert (before==transformer.named_transformers_['numeric'].named_steps['impute'].statistics_).all()
    frame=pd.DataFrame({'customer_id':['a','b','c','d'],'x':[1,7,3,9],'target':['Yes','No','Yes','No'],'answer_copy':['Yes','No','Yes','No']})
    selected,excluded,warnings=choose_features(frame,'target')
    assert selected==['x'];assert any('leakage' in row['reason'] for row in excluded)

def test_queue_claim_cancel_and_retry(workspace):
    _,store,client=workspace;dataset=client.post('/api/datasets/sample').json()
    job=client.post('/api/analyses',json={'dataset_id':dataset['id'],'goal':'Predict churn','time_budget_seconds':120}).json()['id']
    assert store.claim()==job;assert store.claim() is None
    assert client.post(f'/api/analyses/{job}/cancel').status_code==200
    assert store.analysis(job)['cancel_requested']==1
    store.update(job,status='cancelled')
    assert client.post(f'/api/analyses/{job}/retry').status_code==202
    assert store.analysis(job)['cancel_requested']==0


def test_saved_analysis_identifies_its_uploaded_dataset(workspace):
    _,_,client=workspace
    csv='living_area,Price\n'+''.join(f'{50+i},{100000+i*317}\n' for i in range(40))
    dataset=client.post('/api/datasets',files={'file':('homes.csv',csv.encode(),'text/csv')}).json()
    response=client.post('/api/analyses',json={'dataset_id':dataset['id'],'goal':'Predict Price','target':'Price','task':'regression','time_budget_seconds':120})
    assert response.status_code==202
    job=client.get('/api/analyses/'+response.json()['id']).json()
    listed=client.get('/api/analyses').json()[0]
    for row in (job,listed):
        assert row['dataset_filename']=='homes.csv'
        assert row['dataset_id']==dataset['id']
        assert row['request']['target']=='Price'
        assert row['request']['goal']=='Predict Price'
