import numpy as np
import pandas as pd
import pytest
from app.preprocessing.targets import assess_problem, resolve_problem, split_supervised
from app.agents.orchestrator import OrchestratorAgent


def noise_frame(rows=60):
    rng=np.random.default_rng(7)
    return pd.DataFrame({'traffic':rng.normal(size=rows),'decibel_level':rng.uniform(40,95,rows),'sensor_id':np.arange(rows)})


def test_no_last_column_or_identifier_guess_after_sample_replacement():
    data=noise_frame()
    result=assess_problem(data,{'goal':'Predict customer churn and identify factors associated with risk.','task':'classification'})
    assert not result['valid'];assert 'Choose the target column' in result['errors'][0]
    explicit=assess_problem(data,{'goal':'Predict sensor_id','task':'classification','target':'sensor_id'})
    assert not explicit['valid'];assert 'identifier' in explicit['errors'][0]


@pytest.mark.parametrize('as_strings',[False,True])
def test_continuous_measurement_detected_as_regression(as_strings):
    frame=noise_frame()
    if as_strings:frame['decibel_level']=frame['decibel_level'].astype(str)
    result=assess_problem(frame,{'goal':'Predict decibel_level','task':'auto','target':'decibel_level'})
    assert result['valid'];assert result['task']=='regression'
    invalid=assess_problem(frame,{'goal':'Predict decibel_level','task':'classification','target':'decibel_level'})
    assert not invalid['valid'];assert 'Regression' in invalid['errors'][0]


def test_explicit_classification_can_infer_named_target_before_last_column():
    frame=pd.DataFrame({'feature':np.arange(24),'label':['a','b']*12,'sensor_id':np.arange(24)})
    plan=resolve_problem(frame,{'goal':'Classify label','task':'classification'})
    assert plan['target']=='label'


def test_three_row_classes_get_disjoint_holdout_and_two_training_rows():
    frame=pd.DataFrame({'feature':np.arange(24),'label':np.repeat(list('abcdefgh'),3)})
    result=assess_problem(frame,{'goal':'Classify label','task':'classification','target':'label','cv_folds':5})
    assert result['valid'];assert result['cv_folds']==2
    train,test,warnings=split_supervised(frame,'classification','label',42)
    assert warnings;assert train['label'].value_counts().min()==2
    assert test['label'].value_counts().min()==1
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index)|set(test.index)==set(frame.index)


def test_rare_label_error_names_problem_without_dropping_rows():
    frame=pd.DataFrame({'feature':np.arange(22),'label':['common']*20+['rare']*2})
    result=assess_problem(frame,{'goal':'Predict label','task':'classification','target':'label'})
    assert not result['valid'];assert "'rare': 2" in result['errors'][0]
    assert len(frame)==22


def test_validation_endpoint_blocks_invalid_target_before_queue(workspace):
    _,store,client=workspace
    response=client.post('/api/datasets',files={'file':('noise.csv',noise_frame().to_csv(index=False).encode(),'text/csv')})
    dataset=response.json()['id']
    payload={'goal':'Predict customer churn and identify factors associated with risk.','task':'classification'}
    checked=client.post(f'/api/datasets/{dataset}/validate',json=payload)
    assert checked.status_code==200;assert not checked.json()['valid']
    queued=client.post('/api/analyses',json={**payload,'dataset_id':dataset,'time_budget_seconds':120})
    assert queued.status_code==422;assert store.list_analyses()==[]
    corrected=client.post(f'/api/datasets/{dataset}/validate',json={'goal':'Predict decibel_level','target':'decibel_level','task':'auto'})
    assert corrected.json()['valid'];assert corrected.json()['task']=='regression'


def test_small_class_completes_training_and_prediction(workspace):
    settings,store,client=workspace
    rng=np.random.default_rng(11)
    frame=pd.DataFrame({'signal':rng.normal(size=60),'label':['common']*57+['rare']*3})
    dataset=client.post('/api/datasets',files={'file':('rare.csv',frame.to_csv(index=False).encode(),'text/csv')}).json()['id']
    response=client.post('/api/analyses',json={'dataset_id':dataset,'goal':'Predict label','target':'label','task':'classification','cv_folds':5,'quick':True,'tuning_iterations':1,'time_budget_seconds':120,'include_dl':True})
    assert response.status_code==202,response.text
    job=response.json()['id'];assert store.claim()==job
    OrchestratorAgent(settings).run(job)
    record=store.analysis(job);assert record['status']=='completed',record['error']
    assert record['result']['models']['cv_folds']==2
    board=record['result']['models']['leaderboard']
    assert next(row for row in board if row['name']=='neural_network')['status']=='completed'
    assert len(record['result']['evaluation']['classes'])==2
    assert client.post(f'/api/analyses/{job}/predict',json={'rows':[{'signal':0.0}]}).status_code==200
