import test from 'node:test';
import assert from 'node:assert/strict';
import {uploadedDatasetSetup,SAMPLE_GOAL,isIdentifier,analysisGoal,runBlocker,restoredDatasetSetup,analysisTitle,hasLegacySampleGoal} from '../src/analysis-setup.ts';

const uploaded={rows:40,columns:[{name:'decibel_level'},{name:'sensor_id'}]};
test('replacing the churn sample clears its task, target and sample goal',()=>{
  const setup=uploadedDatasetSetup(SAMPLE_GOAL,true);
  assert.deepEqual(setup,{task:'auto',target:'',goal:''});
});
test('first upload preserves a business goal entered before choosing a file',()=>{
  assert.equal(uploadedDatasetSetup('Predict decibel_level').goal,'Predict decibel_level');
});
test('sensor and row identifiers cannot be presented as default outcomes',()=>{
  assert.equal(isIdentifier('sensor_id'),true);
  assert.equal(isIdentifier('id'),true);
  assert.equal(isIdentifier('decibel_level'),false);
});

test('selected Price target runs with an empty business problem',()=>{
  const goal=analysisGoal('','Price','regression');
  assert.equal(goal,'Predict Price and identify the features associated with its predictions.');
  assert.equal(runBlocker({busy:false,connected:true,hasDataset:true,goal,validation:{valid:true,task:'regression',target:'Price',errors:[],warnings:[]},validationError:''}),'');
});
test('custom goals are kept and missing targets are not invented',()=>{
  assert.equal(analysisGoal(' Explain house values ','Price','regression'),'Explain house values');
  assert.equal(analysisGoal('','','auto'),'');
  assert.equal(analysisGoal('','sensor_id','auto'),'');
  assert.equal(analysisGoal('','','clustering'),'Find meaningful groups in this dataset.');
});
test('disabled runs always have an actionable reason',()=>{
  const ready={busy:false,connected:true,hasDataset:true,goal:'Predict Price',validation:{valid:true,task:'regression',target:'Price',errors:[],warnings:[]},validationError:''};
  assert.match(runBlocker({...ready,hasDataset:false}),/Upload/);
  assert.match(runBlocker({...ready,validation:null}),/Checking/);
  assert.equal(runBlocker({...ready,validationError:'Connection timed out'}),'Connection timed out');
  assert.equal(runBlocker({...ready,validation:{...ready.validation,valid:false,errors:['Select an outcome']}}),'Select an outcome');
  assert.match(runBlocker({...ready,goal:'abc'}),/5 characters/);
});

test('replacing any dataset clears its previous custom goal',()=>{
  assert.equal(uploadedDatasetSetup('Predict Price',true).goal,'');
});

test('reopening an old run clears sample text and retains its explicit target',()=>{
  const request={goal:SAMPLE_GOAL,target:'decibel_level',task:'regression'};
  assert.deepEqual(restoredDatasetSetup(uploaded as never,request as never),{task:'auto',target:'decibel_level',goal:''});
  const custom={...request,goal:'Estimate urban noise levels'};
  assert.equal(restoredDatasetSetup(uploaded as never,custom as never).goal,custom.goal);
});

test('reopened generated goals follow a changed target',()=>{
  const request={goal:analysisGoal('','decibel_level','regression'),target:'decibel_level',task:'regression'};
  const setup=restoredDatasetSetup(uploaded as never,request as never);
  assert.equal(setup.goal,'');
  assert.match(analysisGoal(setup.goal,'Price','regression'),/^Predict Price /);
});

test('saved labels reflect the measured task and target, not a stale sample goal',()=>{
  const job={request:{goal:SAMPLE_GOAL,target:'Price',task:'auto'},result:{plan:{task:'regression',target:'Price'}}} as never;
  assert.equal(analysisTitle(job),'Regression · Price');
  assert.equal(hasLegacySampleGoal(job),true);
  assert.equal(analysisTitle({request:{goal:SAMPLE_GOAL,target:null,task:'auto'},result:null} as never),'Analysis · target not selected');
  assert.equal(hasLegacySampleGoal({request:{goal:SAMPLE_GOAL,target:'churn',task:'classification'},result:null} as never),false);
});
