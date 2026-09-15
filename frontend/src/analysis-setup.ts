import type {AnalysisInput,Dataset,Job,ProblemValidation} from './types';

export const SAMPLE_GOAL='Predict customer churn and identify factors associated with risk.';
export const isIdentifier=(name:string)=>/(^id$|_id$|^index$|^unnamed:|^name$|email|phone)/i.test(name);

export function analysisGoal(goal:string,target:string,task:string){
  if(goal.trim())return goal.trim();
  if(task==='clustering')return 'Find meaningful groups in this dataset.';
  return target&&!isIdentifier(target)?`Predict ${target} and identify the features associated with its predictions.`:'';
}

export function runBlocker({busy,connected,hasDataset,goal,validation,validationError}:{busy:boolean;connected:boolean;hasDataset:boolean;goal:string;validation:ProblemValidation|null;validationError:string}){
  if(busy)return 'Working…';
  if(!connected)return 'Connect to the Python backend to start an analysis.';
  if(!hasDataset)return 'Upload a dataset or try the sample first.';
  if(validationError)return validationError;
  if(!validation)return 'Checking the selected target…';
  if(!validation.valid)return validation.errors[0]||'Review the target and problem type.';
  if(!goal)return 'Select a target or describe the outcome you want to predict.';
  if(goal.length<5)return 'Use at least 5 characters for a custom business problem, or leave it blank to use the selected target.';
  return '';
}

const isSampleGoal=(goal:string)=>/^predict customer churn and identify (factors associated with risk|associated risk factors)\.?$/i.test(goal.trim());

export function uploadedDatasetSetup(previousGoal:string,replacingDataset=false){
  return {task:'auto',target:'',goal:replacingDataset||isSampleGoal(previousGoal)?'':previousGoal};
}

export function restoredDatasetSetup(dataset:Dataset,request:AnalysisInput){
  const target=request.target&&dataset.columns.some(c=>c.name===request.target)&&!isIdentifier(request.target)?request.target:'';
  const generated=analysisGoal('',target,request.task);
  return {task:isSampleGoal(request.goal)?'auto':request.task,target,goal:isSampleGoal(request.goal)||request.goal===generated?'':request.goal};
}

export function analysisTitle(job:Pick<Job,'request'|'result'>){
  const task=job.result?.plan.task||job.request.task;
  const target=job.result?.plan.target||job.request.target;
  if(task==='clustering')return 'Clustering';
  if(target&&!isIdentifier(target))return `${task==='classification'?'Classification':task==='regression'?'Regression':'Prediction'} · ${target}`;
  return 'Analysis · target not selected';
}

export function hasLegacySampleGoal(job:Pick<Job,'request'|'result'>){
  const target=job.result?.plan.target||job.request.target;
  return isSampleGoal(job.request.goal)&&target?.toLowerCase()!=='churn';
}
