import json
import time
import httpx
import pytest
from app.settings import Settings
from app.tools.llm import LLMClient
from app.worker import supervise
from app.tools.data import export_csv
import pandas as pd

def test_llm_retries_only_reasoning_json(monkeypatch):
    settings=Settings(llm_base_url='https://llm.example/v1',llm_api_key='test-only',llm_model='test-model')
    seen=[]
    def handler(request):
        seen.append(json.loads(request.content))
        if len(seen)==1:return httpx.Response(503,json={'error':'temporary'})
        return httpx.Response(200,json={'choices':[{'message':{'content':'{"summary":"Evidence only","assumptions":[]}'}}]})
    original=httpx.Client
    monkeypatch.setattr('app.tools.llm.httpx.Client',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    result=LLMClient(settings).reason('Explain the plan.',{'rows':600,'columns':['tenure','churn']})
    assert result['summary']=='Evidence only';assert len(seen)==2
    assert 'tools' not in seen[0];assert 'rows' in seen[0]['messages'][1]['content']

def test_upload_body_limit_is_enforced(workspace):
    settings,_,client=workspace
    response=client.post('/api/datasets',content=b'x'*(settings.max_upload_mb*1024**2+65537),headers={'Content-Type':'application/octet-stream'})
    assert response.status_code==413

class FakeProcess:
    pid=987654321
    exitcode=None
    def start(self):pass
    def is_alive(self):return True
    def join(self,timeout=None):pass

def test_supervisor_time_limit_marks_job_and_keeps_checkpoints(workspace,monkeypatch):
    settings,store,client=workspace;data=client.post('/api/datasets/sample').json()
    job=client.post('/api/analyses',json={'dataset_id':data['id'],'goal':'Predict churn','time_budget_seconds':10}).json()['id'];store.claim()
    store.checkpoint(job,'profiling',{'rows':600},.1)
    process=FakeProcess();stopped=[]
    monkeypatch.setattr('app.worker.multiprocessing.get_context',lambda _:type('Context',(),{'Process':lambda *a,**k:process})())
    clock=iter([0,11]);monkeypatch.setattr('app.worker.time.monotonic',lambda:next(clock))
    monkeypatch.setattr('app.worker.stop_process',lambda p:stopped.append(p))
    supervise(job,settings,store)
    assert store.analysis(job)['status']=='timed_out';assert stopped==[process]
    assert store.analysis(job)['checkpoints'][0]['stage']=='profiling'

def test_worker_recovery_requires_expired_heartbeat(workspace):
    _,store,client=workspace;data=client.post('/api/datasets/sample').json()
    job=client.post('/api/analyses',json={'dataset_id':data['id'],'goal':'Predict churn','time_budget_seconds':120}).json()['id'];store.claim();store.recover()
    assert store.analysis(job)['status']=='running'
    store.update(job,heartbeat=time.time()-60);store.recover()
    assert store.analysis(job)['status']=='interrupted'


def test_download_csv_neutralizes_formulas_without_changing_training_data(tmp_path):
    frame=pd.DataFrame({'=header':['=1+1',' @SUM(A1)','ordinary'], 'number':[-2,3,4]})
    destination=tmp_path/'cleaned.csv';export_csv(frame,destination)
    exported=pd.read_csv(destination)
    assert exported.columns[0]=="'=header"
    assert exported.iloc[0,0]=="'=1+1";assert exported.iloc[1,0]=="' @SUM(A1)"
    assert frame.iloc[0,0]=='=1+1';assert exported['number'].tolist()==[-2,3,4]


def test_retry_cannot_bypass_queue_limit(workspace):
    _,store,client=workspace;dataset=client.post('/api/datasets/sample').json()['id']
    first=client.post('/api/analyses',json={'dataset_id':dataset,'goal':'Predict churn','time_budget_seconds':120}).json()['id']
    client.post(f'/api/analyses/{first}/cancel')
    second=client.post('/api/analyses',json={'dataset_id':dataset,'goal':'Predict churn again','time_budget_seconds':120})
    assert second.status_code==202,second.text
    with pytest.raises(ValueError,match='queue is full'):store.retry(first,max_queue=1)
    assert store.analysis(first)['status']=='cancelled'
