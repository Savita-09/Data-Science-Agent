import test from 'node:test';
import assert from 'node:assert/strict';
import {ApiError, BackendUnavailableError, requestJson, waitForBackend} from '../src/transport.ts';

test('cold start recovers from network failure, gateway error, and loading HTML',async()=>{
  let calls=0,retries=0;
  const fetcher:typeof fetch=async(_url,options)=>{
    assert.equal(options?.headers,undefined,'health checks must not send an API key');
    assert.equal(options?.credentials,'omit');
    calls++;
    if(calls===1)throw new TypeError('Failed to fetch');
    if(calls===2)return new Response('Starting',{status:503});
    if(calls===3)return new Response('<html>Waking up</html>',{headers:{'content-type':'text/html'}});
    return Response.json({status:'ok'});
  };
  const health=await waitForBackend('https://api.example.com/api/health',{
    signal:new AbortController().signal,fetcher,retryMs:1,timeoutMs:1000,onRetry:()=>retries++,
  });
  assert.deepEqual(health,{status:'ok'});
  assert.equal(calls,4);assert.equal(retries,3);
});

test('authentication and configuration errors fail immediately instead of retrying',async()=>{
  for(const status of [401,403,404,422]){
    let calls=0;
    await assert.rejects(waitForBackend('https://api.example.com/api/health',{
      signal:new AbortController().signal,retryMs:1,
      fetcher:async()=>{calls++;return Response.json({detail:'Check configuration'},{status});},
    }),error=>error instanceof ApiError&&error.status===status);
    assert.equal(calls,1);
  }
});

test('reconnection has a deadline even when the server never responds',async()=>{
  let calls=0;
  const started=Date.now();
  await assert.rejects(waitForBackend('https://api.example.com/api/health',{
    signal:new AbortController().signal,timeoutMs:35,attemptMs:8,retryMs:1,
    fetcher:async(_url,options)=>{
      calls++;
      return new Promise((_resolve,reject)=>options!.signal!.addEventListener('abort',()=>reject(options!.signal!.reason),{once:true}));
    },
  }),BackendUnavailableError);
  assert.ok(calls>=1);assert.ok(Date.now()-started<1000);
});

test('closing a page cancels reconnection and further attempts',async()=>{
  const controller=new AbortController();let calls=0;
  const pending=waitForBackend('https://api.example.com/api/health',{
    signal:controller.signal,retryMs:1000,
    fetcher:async()=>{calls++;throw new TypeError('offline');},
    onRetry:()=>controller.abort(new Error('Page closed')),
  });
  await assert.rejects(pending,/Page closed/);assert.equal(calls,1);
});

test('training submissions are never replayed after an ambiguous network failure',async()=>{
  let calls=0;
  await assert.rejects(requestJson('https://api.example.com/api/analyses',{
    method:'POST',body:JSON.stringify({dataset_id:'example'}),
  },async()=>{calls++;throw new TypeError('connection lost');}),BackendUnavailableError);
  assert.equal(calls,1);
});
