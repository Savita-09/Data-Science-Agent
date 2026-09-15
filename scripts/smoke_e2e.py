"""Live HTTP verification against a running API + worker, using synthetic data only."""
import argparse
import json
import os
import time
from pathlib import Path
import httpx

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8000');parser.add_argument('--output',default='test-results/live-e2e.json');args=parser.parse_args()
    headers={'X-API-Key':os.environ['APP_API_KEY']} if os.environ.get('APP_API_KEY') else {}
    with httpx.Client(base_url=args.url,headers=headers,timeout=30) as client:
        assert client.get('/api/health').is_success
        response=client.post('/api/datasets/sample');response.raise_for_status();dataset=response.json()
        response=client.post('/api/analyses',json={'dataset_id':dataset['id'],'goal':'Predict customer churn and identify associated risk factors.','target':'churn','task':'auto','include_dl':True,'quick':True,'cv_folds':3,'tuning_iterations':1,'time_budget_seconds':180});response.raise_for_status();analysis_id=response.json()['id']
        started=time.monotonic()
        while True:
            response=client.get(f'/api/analyses/{analysis_id}');response.raise_for_status();job=response.json()
            if job['status'] not in ('queued','running'):break
            if time.monotonic()-started>210:raise TimeoutError('Live workflow did not finish in time.')
            time.sleep(1)
        assert job['status']=='completed',job['error']
        result=job['result'];row={column['name']:column['example'] for column in result['model_schema']}
        prediction=client.post(f'/api/analyses/{analysis_id}/predict',json={'rows':[row]});prediction.raise_for_status()
        artifacts={}
        for kind in ('html','pdf','json','model','predictions','cleaned'):
            response=client.get(f'/api/analyses/{analysis_id}/artifacts/{kind}');response.raise_for_status();artifacts[kind]=len(response.content)
        answer=client.post(f'/api/analyses/{analysis_id}/chat',json={'question':'Which features matter?'});answer.raise_for_status()
        summary={'analysis_id':analysis_id,'dataset_rows':dataset['rows'],'status':job['status'],'seconds':round(time.monotonic()-started,2),'checkpoints':len(job['checkpoints']),'winner':result['models']['winner'],'metrics':result['evaluation']['metrics'],'baseline':result['evaluation']['baseline'],'shap_method':result['explainability'].get('method'),'max_shap_additivity_error':max((r['additivity_error'] for r in result['explainability'].get('local',[])),default=0),'prediction':prediction.json(),'artifact_bytes':artifacts,'chat_source':answer.json()['source']}
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
