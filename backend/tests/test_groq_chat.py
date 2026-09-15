import json
import httpx
import pytest
from app.settings import Settings, get_settings
from app.tools.groq_chat import GroqChatClient, GroqChatError, GROQ_CHAT_URL


def intercept(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr('app.tools.groq_chat.httpx.Client', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr('app.tools.groq_chat.time.sleep', lambda seconds: None)


def complete_job(workspace):
    settings,store,client=workspace
    dataset=client.post('/api/datasets/sample').json()
    response=client.post('/api/analyses',json={'dataset_id':dataset['id'],'goal':'Predict churn','target':'churn','time_budget_seconds':120})
    job=response.json()['id']
    store.update(job,status='completed',result={
        'plan':{'task':'classification','target':'churn'},
        'models':{'winner':'linear'},
        'evaluation':{'metrics':{'accuracy':.8},'baseline':{'accuracy':.6},'test_rows':120,'warnings':['Associations are not causal.'],'predictions':[{'private':'PRIVATE_ROW_SECRET'}]},
        'explainability':{'global':[{'feature':'tenure','mean_abs_shap':.1}],'local':[{'private':'PRIVATE_ROW_SECRET'}]},
        'profile':{'preview':[{'private':'PRIVATE_ROW_SECRET'}]},
        'insights':{'summary':'Measured accuracy is 0.8.','recommendations':[]},
    })
    return settings,store,client,job


def test_groq_env_and_health_do_not_expose_the_key(monkeypatch,workspace):
    monkeypatch.setenv('GROQ_API_KEY','test-key-not-a-real-credential')
    monkeypatch.setenv('GROQ_MODEL','openai/gpt-oss-20b')
    configured=get_settings()
    assert configured.groq_configured
    assert configured.groq_model=='openai/gpt-oss-20b'
    assert 'test-key-not-a-real-credential' not in repr(configured)
    settings,_,client=workspace
    settings.groq_api_key=configured.groq_api_key
    health=client.get('/api/health')
    assert health.json()['groq_configured'] is True
    assert 'test-key-not-a-real-credential' not in health.text


def test_groq_chat_uses_aggregates_and_history_and_persists_reply(workspace,monkeypatch):
    settings,store,client,job=complete_job(workspace)
    settings.groq_api_key='test-groq-key'
    store.save_chat(job,'Which model won?','The linear model.','local')
    seen=[]
    def handler(request):
        assert str(request.url)==GROQ_CHAT_URL
        assert request.headers['Authorization']=='Bearer test-groq-key'
        payload=json.loads(request.content)
        assert payload['model']==settings.groq_model
        assert payload['messages'][-1]['content']=='Why was that selected?'
        assert any(m['content']=='The linear model.' for m in payload['messages'])
        assert 'PRIVATE_ROW_SECRET' not in request.content.decode()
        assert '"accuracy": 0.8' in payload['messages'][1]['content']
        assert 'tools' not in payload and 'functions' not in payload
        seen.append(payload)
        return httpx.Response(200,json={'choices':[{'message':{'content':'The selected model had the strongest validation performance.'},'finish_reason':'stop'}]})
    intercept(monkeypatch,handler)
    response=client.post(f'/api/analyses/{job}/chat',json={'question':'Why was that selected?','provider':'groq'})
    assert response.status_code==200,response.text
    assert response.json()['source']=='Groq · openai/gpt-oss-20b'
    assert len(seen)==1 and len(store.chats(job))==2
    assert store.chats(job)[-1]['answer']==response.json()['answer']


def test_missing_groq_key_does_not_call_provider_or_store_fake_reply(workspace,monkeypatch):
    settings,store,client,job=complete_job(workspace)
    settings.groq_api_key=''
    intercept(monkeypatch,lambda request:pytest.fail('Missing credentials must not reach Groq'))
    response=client.post(f'/api/analyses/{job}/chat',json={'question':'Explain the result','provider':'groq'})
    assert response.status_code==422 and 'GROQ_API_KEY' in response.text
    assert store.chats(job)==[]
    local=client.post(f'/api/analyses/{job}/chat',json={'question':'Summarize performance','provider':'local'})
    assert local.status_code==200 and local.json()['source']=='deterministic report assistant'
    assert client.post(f'/api/analyses/{job}/chat',json={'question':'    ','provider':'groq'}).status_code==422


@pytest.mark.parametrize('provider_status,attempts,client_status',[(401,1,502),(403,1,502),(404,1,502),(429,3,429),(503,3,503)])
def test_provider_errors_are_sanitized_and_retries_are_bounded(monkeypatch,provider_status,attempts,client_status):
    seen=[]
    def handler(request):
        seen.append(request)
        return httpx.Response(provider_status,json={'error':{'message':'secret-test-groq-key'}},headers={'Retry-After':'0'})
    intercept(monkeypatch,handler)
    with pytest.raises(GroqChatError) as raised:
        GroqChatClient(Settings(groq_api_key='secret-test-groq-key')).answer('Explain this',{})
    assert raised.value.status_code==client_status
    assert 'secret-test-groq-key' not in str(raised.value)
    assert len(seen)==attempts


def test_transient_rate_limit_recovers_without_duplicate_answers(monkeypatch):
    seen=[]
    def handler(request):
        seen.append(request)
        if len(seen)==1:return httpx.Response(429,headers={'Retry-After':'0'})
        return httpx.Response(200,json={'choices':[{'message':{'content':'Recovered reply.'}}]})
    intercept(monkeypatch,handler)
    assert GroqChatClient(Settings(groq_api_key='test-only')).answer('Explain this',{})=='Recovered reply.'
    assert len(seen)==2


@pytest.mark.parametrize('body',[
    {'choices':[]},
    {'choices':[{'message':{'content':None}}]},
    {'choices':[{'message':{'content':' '}}]},
    {'choices':[{'message':{'content':'Partial reply'},'finish_reason':'length'}]},
])
def test_invalid_or_truncated_replies_are_not_presented_as_success(monkeypatch,body):
    intercept(monkeypatch,lambda request:httpx.Response(200,json=body))
    with pytest.raises(GroqChatError):
        GroqChatClient(Settings(groq_api_key='test-only')).answer('Explain this',{})


def test_timeout_is_bounded_and_history_is_limited(monkeypatch):
    seen=[]
    def handler(request):
        payload=json.loads(request.content)
        assert len(payload['messages'])==11  # system, facts, four turns, latest question
        seen.append(request)
        raise httpx.ReadTimeout('test timeout',request=request)
    intercept(monkeypatch,handler)
    with pytest.raises(GroqChatError) as raised:
        GroqChatClient(Settings(groq_api_key='test-only')).answer('Explain this',{},[{'question':'x'*3000,'answer':'y'*6000}]*20)
    assert raised.value.status_code==503 and len(seen)==3
