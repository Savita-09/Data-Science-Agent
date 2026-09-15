import asyncio
import hashlib
import hmac
import json
import os
from uuid import UUID,uuid4
import joblib
import pandas as pd
from fastapi import FastAPI,Depends,UploadFile,File,HTTPException,Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,StreamingResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from app.settings import get_settings,ROOT
from app.database.store import Store
from app.api.schemas import AnalysisRequest,PredictionRequest,ChatRequest,ProblemRequest
from app.tools.data import read_upload,describe,synthetic_churn,fingerprint,json_safe
from app.preprocessing.pipeline import normalize_prediction
from app.preprocessing.targets import assess_problem
from app.tools.llm import LLMClient
from app.tools.groq_chat import GroqChatClient, GroqChatError, analysis_facts
from app.tools.dashboard import dataset_dashboard, dataset_relationship
from app.reporting.reports import prepare_report, render_html, render_pdf

class BodyLimitMiddleware:
    def __init__(self,app,limit):self.app=app;self.limit=limit
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['method'] not in ('POST','PUT','PATCH'):return await self.app(scope,receive,send)
        chunks=[];total=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            data=message.get('body',b'');total+=len(data)
            if total>self.limit:
                return await JSONResponse({'detail':'Request body exceeds the configured size limit.'},status_code=413)(scope,receive,send)
            chunks.append(data)
            if not message.get('more_body',False):break
        body=b''.join(chunks);sent=False
        async def replay():
            nonlocal sent
            if not sent:sent=True;return {'type':'http.request','body':body,'more_body':False}
            return await receive()
        await self.app(scope,replay,send)

def create_app(settings=None):
    settings=settings or get_settings();settings.initialize();store=Store(settings.db_path);store.initialize()
    app=FastAPI(title='Autonomous AI Data Scientist API',version='1.0.0',description='Durable analysis workflows. Numeric computations run in Python; the optional LLM supplies narrative reasoning only.')
    app.state.settings=settings;app.state.store=store
    app.add_middleware(CORSMiddleware,allow_origins=settings.cors_origins,allow_methods=['GET','POST'],allow_headers=['Content-Type','X-API-Key'],expose_headers=['Content-Disposition'])
    app.add_middleware(BodyLimitMiddleware,limit=settings.max_upload_mb*1024**2+65536)
    key_header=APIKeyHeader(name='X-API-Key',auto_error=False)

    def authorized(request:Request,key:str|None=Depends(key_header)):
        if settings.api_key:
            if not key or not hmac.compare_digest(key,settings.api_key):raise HTTPException(401,'A valid X-API-Key header is required.')
        elif request.client and request.client.host not in ('127.0.0.1','::1','localhost','testclient'):
            raise HTTPException(403,'Unauthenticated development access is restricted to the local computer.')

    def analysis_or_404(analysis_id):
        row=store.analysis(str(analysis_id))
        if not row:raise HTTPException(404,'Analysis not found.')
        return row

    @app.middleware('http')
    async def headers(request,call_next):
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        if request.url.path.startswith('/api'):response.headers['Cache-Control']='no-store'
        return response

    @app.get('/api/health')
    def health():return {'status':'ok','version':'1.0.0','llm_configured':settings.llm_configured,'groq_configured':settings.groq_configured,'groq_model':settings.groq_model,'auth_required':bool(settings.api_key),'ephemeral_storage':settings.ephemeral_storage,'max_rows':settings.max_rows,'max_upload_mb':settings.max_upload_mb,'max_job_seconds':settings.max_job_seconds}

    def persist_dataset(frame,filename):
        dataset_id=str(uuid4());content=frame.to_csv(index=False).encode('utf-8');sha=hashlib.sha256(content).hexdigest()
        path=settings.data_root/(dataset_id+'.csv');path.write_bytes(content)
        metadata=describe(frame)
        try:store.add_dataset(dataset_id,filename,sha,metadata)
        except Exception:path.unlink(missing_ok=True);raise
        return {'id':dataset_id,'filename':filename,**metadata}

    @app.post('/api/datasets',dependencies=[Depends(authorized)],status_code=201)
    async def upload(file:UploadFile=File(...)):
        try:
            content=await file.read(settings.max_upload_mb*1024**2+1)
            frame=await run_in_threadpool(read_upload,content,file.filename or 'dataset.csv',settings)
            return await run_in_threadpool(persist_dataset,frame,os.path.basename(file.filename or 'dataset.csv'))
        except (ValueError,UnicodeError) as error:raise HTTPException(422,str(error)) from error
        finally:await file.close()

    @app.post('/api/datasets/sample',dependencies=[Depends(authorized)],status_code=201)
    def sample():return persist_dataset(synthetic_churn(),'synthetic-customer-churn.csv')

    @app.get('/api/datasets',dependencies=[Depends(authorized)])
    def datasets():
        with store.connect() as db:
            rows=db.execute('SELECT id,filename,metadata,created FROM datasets ORDER BY created DESC LIMIT 100').fetchall()
        return [{'id':row['id'],'filename':row['filename'],'created':row['created'],'rows':json.loads(row['metadata'])['rows'],'column_count':json.loads(row['metadata'])['column_count']} for row in rows]

    @app.get('/api/datasets/{dataset_id}',dependencies=[Depends(authorized)])
    def dataset(dataset_id:UUID):
        row=store.dataset(str(dataset_id))
        if not row:raise HTTPException(404,'Dataset not found.')
        return {'id':row['id'],'filename':row['filename'],**row['metadata']}

    @app.get('/api/datasets/{dataset_id}/eda',dependencies=[Depends(authorized)])
    def dashboard(dataset_id:UUID,target:str|None=None):
        row=store.dataset(str(dataset_id))
        if not row:raise HTTPException(404,'Dataset not found.')
        frame=pd.read_csv(settings.data_root/(str(dataset_id)+'.csv'))
        if target is not None and target not in frame:raise HTTPException(422,'Unknown target column.')
        return {'id':row['id'],'filename':row['filename'],**dataset_dashboard(frame,target)}

    @app.get('/api/datasets/{dataset_id}/relationship',dependencies=[Depends(authorized)])
    def relationship(dataset_id:UUID,x:str,y:str):
        if not store.dataset(str(dataset_id)):raise HTTPException(404,'Dataset not found.')
        frame=pd.read_csv(settings.data_root/(str(dataset_id)+'.csv'))
        try:return dataset_relationship(frame,x,y)
        except ValueError as error:raise HTTPException(422,str(error)) from error

    @app.post('/api/analyses',dependencies=[Depends(authorized)],status_code=202)
    def start(body:AnalysisRequest):
        dataset=store.dataset(str(body.dataset_id))
        if not dataset:raise HTTPException(404,'Dataset not found.')
        if body.time_budget_seconds>settings.max_job_seconds:raise HTTPException(422,f'Time budget exceeds server limit of {settings.max_job_seconds} seconds.')
        if body.tuning_iterations>settings.tuning_iterations:raise HTTPException(422,f'Tuning is limited to {settings.tuning_iterations} configurations per model.')
        if body.use_llm and not settings.llm_configured:raise HTTPException(422,'Configure the LLM environment variables or run without LLM reasoning.')
        columns={c['name'] for c in dataset['metadata']['columns']}
        if body.target and body.target not in columns:raise HTTPException(422,'Unknown target column.')
        request=body.model_dump(mode='json');analysis_id=str(uuid4())
        assessment=assess_problem(pd.read_csv(settings.data_root/(str(body.dataset_id)+'.csv')),request)
        if not assessment['valid']:raise HTTPException(422,' '.join(assessment['errors']))
        try:store.enqueue(analysis_id,str(body.dataset_id),request,fingerprint(dataset['sha'],request),settings.max_queued_jobs)
        except ValueError as error:raise HTTPException(429,str(error)) from error
        return {'id':analysis_id,'status':'queued','status_url':f'/api/analyses/{analysis_id}','events_url':f'/api/analyses/{analysis_id}/events'}

    @app.post('/api/datasets/{dataset_id}/validate',dependencies=[Depends(authorized)])
    def validate_problem(dataset_id:UUID,body:ProblemRequest):
        if not store.dataset(str(dataset_id)):raise HTTPException(404,'Dataset not found.')
        return assess_problem(pd.read_csv(settings.data_root/(str(dataset_id)+'.csv')),body.model_dump())

    @app.get('/api/analyses',dependencies=[Depends(authorized)])
    def listing():
        return [{**row,'request':json.loads(row['request'])} for row in store.list_analyses()]

    @app.get('/api/analyses/{analysis_id}',dependencies=[Depends(authorized)])
    def get_analysis(analysis_id:UUID):
        row=analysis_or_404(analysis_id)
        # Full prediction exports are separate; keep progress responses bounded.
        if row['result']:
            evaluation=row['result']['evaluation'];evaluation['prediction_count']=len(evaluation['predictions']);evaluation['predictions']=evaluation['predictions'][:50]
        for cp in row['checkpoints']:
            if cp['stage']=='evaluation':cp['result']['predictions']=cp['result'].get('predictions',[])[:50]
        return row

    @app.get('/api/analyses/{analysis_id}/events',dependencies=[Depends(authorized)])
    async def events(analysis_id:UUID,request:Request,after:int=0):
        analysis_or_404(analysis_id)
        async def stream():
            cursor=max(0,after)
            while not await request.is_disconnected():
                rows=await run_in_threadpool(store.events,str(analysis_id),cursor)
                for row in rows:
                    cursor=row['id'];yield f'id: {cursor}\nevent: progress\ndata: {json.dumps(row)}\n\n'
                state=await run_in_threadpool(store.analysis,str(analysis_id))
                if state['status'] not in ('queued','running'):
                    yield f'event: done\ndata: {json.dumps({"status":state["status"]})}\n\n';return
                yield ': heartbeat\n\n';await asyncio.sleep(1)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'X-Accel-Buffering':'no'})

    @app.post('/api/analyses/{analysis_id}/cancel',dependencies=[Depends(authorized)])
    def cancel(analysis_id:UUID):
        row=analysis_or_404(analysis_id)
        if row['status']=='queued':store.update(str(analysis_id),status='cancelled',cancel_requested=1)
        elif row['status']=='running':store.update(str(analysis_id),cancel_requested=1)
        else:raise HTTPException(409,'Only queued or running analyses can be cancelled.')
        return {'status':'cancellation_requested'}

    @app.post('/api/analyses/{analysis_id}/retry',dependencies=[Depends(authorized)],status_code=202)
    def retry(analysis_id:UUID):
        row=analysis_or_404(analysis_id)
        if row['status'] in ('failed','cancelled','timed_out','interrupted'):
            assessment=assess_problem(pd.read_csv(settings.data_root/(row['dataset_id']+'.csv')),row['request'])
            if not assessment['valid']:raise HTTPException(422,'Edit the analysis setup before retrying. '+' '.join(assessment['errors']))
        try:store.retry(str(analysis_id),settings.max_queued_jobs)
        except ValueError as error:raise HTTPException(409,str(error)) from error
        return {'id':str(analysis_id),'status':'queued','resume':'completed checkpoints'}

    @app.get('/api/analyses/{analysis_id}/artifacts/{kind}',dependencies=[Depends(authorized)])
    def artifact(analysis_id:UUID,kind:str):
        row=analysis_or_404(analysis_id)
        names={'html':('report.html','text/html'),'pdf':('report.pdf','application/pdf'),'json':('report.json','application/json'),'model':('model.joblib','application/octet-stream'),'predictions':('predictions.csv','text/csv'),'cleaned':('cleaned.csv','text/csv')}
        if kind not in names:raise HTTPException(404,'Artifact not found.')
        name,mime=names[kind];path=settings.project_root/str(analysis_id)/name
        if row['status']!='completed':raise HTTPException(409,'Artifacts are available after the analysis completes.')
        if kind in ('html','pdf','json'):
            # Render current report layout for existing runs; never retrain or rewrite saved evidence.
            original=None if 'report_eda' in row['result'] else pd.read_csv(settings.data_root/(row['dataset_id']+'.csv'))
            result=prepare_report(row['result'],filename=row['dataset_filename'],request=row['request'],frame=original)
            content=render_html(result) if kind=='html' else render_pdf(result) if kind=='pdf' else json.dumps(result,indent=2,allow_nan=False)
            return Response(content=content,media_type=mime,headers={'Content-Disposition':f'attachment; filename="{analysis_id}-{name}"'})
        if not path.is_file():raise HTTPException(409,'Artifact file is unavailable.')
        return FileResponse(path,media_type=mime,filename=f'{analysis_id}-{name}')

    @app.post('/api/analyses/{analysis_id}/predict',dependencies=[Depends(authorized)])
    def predict(analysis_id:UUID,body:PredictionRequest):
        row=analysis_or_404(analysis_id)
        if row['status']!='completed':raise HTTPException(409,'Complete the analysis before predicting.')
        bundle=joblib.load(settings.project_root/str(analysis_id)/'model.joblib')
        try:
            frame=normalize_prediction(body.rows,bundle['feature_schema']);predicted=bundle['pipeline'].predict(frame)
            values=[bundle['classes'][int(v)] for v in predicted] if bundle['task']=='classification' else predicted.tolist()
            result={'predictions':values,'model':bundle['winner'],'task':bundle['task']}
            if bundle['task']=='classification':result['probabilities']=bundle['pipeline'].predict_proba(frame).tolist();result['classes']=bundle['classes']
            return json_safe(result)
        except ValueError as error:raise HTTPException(422,str(error)) from error

    @app.get('/api/analyses/{analysis_id}/chat',dependencies=[Depends(authorized)])
    def chat_history(analysis_id:UUID):analysis_or_404(analysis_id);return store.chats(str(analysis_id))

    @app.post('/api/analyses/{analysis_id}/chat',dependencies=[Depends(authorized)])
    def chat(analysis_id:UUID,body:ChatRequest):
        row=analysis_or_404(analysis_id)
        if row['status']!='completed':raise HTTPException(409,'Complete the analysis before discussing its results.')
        result=row['result'];source='deterministic report assistant'
        provider=body.provider or ('configured' if body.use_llm else 'local')
        if provider=='groq':
            try:
                answer=GroqChatClient(settings).answer(body.question,analysis_facts(row),store.chats(str(analysis_id)))
                source=f'Groq · {settings.groq_model}'
            except GroqChatError as error:raise HTTPException(error.status_code,str(error)) from error
        elif provider=='configured':
            if not settings.llm_configured:raise HTTPException(422,'LLM is not configured.')
            try:
                response=LLMClient(settings).reason('Answer the question using these analysis facts. Return {"answer":string}. If the facts cannot answer it, say so.',{'question':body.question,'goal':row['request']['goal'],'metrics':result['evaluation']['metrics'],'warnings':result['evaluation']['warnings'],'model':result['models']['winner'],'important_features':result['explainability'].get('global',[])[:10]})
                if not isinstance(response.get('answer'),str):raise ValueError('Invalid answer from LLM.')
                answer=response['answer'][:6000];source='configured LLM'
            except ValueError as error:raise HTTPException(502,str(error)) from error
        else:
            question=body.question.lower()
            if any(word in question for word in ('feature','shap','why','explain')):
                top=[v['feature'] for v in result['explainability'].get('global',[])[:5]]
                answer='The largest measured SHAP contributions are associated with '+', '.join(top)+'. These are associations, not causal drivers.' if top else result['explainability'].get('reason','No SHAP explanation is available.')
            elif any(word in question for word in ('improve','recommend','next','business')):answer=' '.join(result['insights']['recommendations'])
            else:answer=result['insights']['summary']+' This local report assistant can summarize model performance, SHAP, and recommended next steps. Enable the configured LLM for open-ended reasoning.'
        store.save_chat(str(analysis_id),body.question,answer,source)
        return {'answer':answer,'source':source}

    dist=ROOT/'frontend'/'dist'
    if dist.is_dir():
        app.mount('/assets',StaticFiles(directory=dist/'assets'),name='assets')
        @app.get('/{path:path}',include_in_schema=False)
        def frontend(path:str):
            if path not in ('','index.html'):raise HTTPException(404,'Endpoint not found.')
            return FileResponse(dist/'index.html')
    return app

app=create_app()
