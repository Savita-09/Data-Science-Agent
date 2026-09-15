import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeApiBase} from '../src/api-config.ts';

test('local combined app uses the same origin',()=>assert.equal(normalizeApiBase(''),'/api'));
test('remote backend URLs keep their path and do not duplicate api',()=>{
  assert.equal(normalizeApiBase(' https://api.example.com/ ','https:'),'https://api.example.com/api');
  assert.equal(normalizeApiBase('https://api.example.com/api///','https:'),'https://api.example.com/api');
  assert.equal(normalizeApiBase('https://example.com/backend','https:'),'https://example.com/backend/api');
});
test('unsafe addresses and embedded credentials are rejected before sending an API key',()=>{
  for(const value of ['javascript:alert(1)','ftp://example.com','https://user:password@example.com','https://example.com?token=x','https://example.com#x','/relative']){
    assert.throws(()=>normalizeApiBase(value));
  }
});
test('an HTTPS dashboard requires an HTTPS backend',()=>{
  assert.throws(()=>normalizeApiBase('http://api.example.com','https:'),/HTTPS/);
  assert.equal(normalizeApiBase('http://127.0.0.1:8000','http:'),'http://127.0.0.1:8000/api');
});
