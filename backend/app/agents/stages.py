import pandas as pd
from sklearn.preprocessing import LabelEncoder
from app.tools.data import describe, eda, json_safe, export_csv
from app.tools.llm import LLMClient
from app.preprocessing.pipeline import clean_frame, choose_features
from app.preprocessing.targets import resolve_problem, validate_target, split_supervised
from app.ml.training import train_models, evaluate, saved_bundle, atomic_joblib
from app.explainability.shap_tools import explain
from app.reporting.reports import build_reports, prepare_report

class Agent:
    id='';title=''
    def run(self, ctx, settings, path, event): raise NotImplementedError

class ProblemUnderstandingAgent(Agent):
    id='understanding';title='Problem Understanding'
    def run(self,ctx,settings,path,event):
        result=resolve_problem(clean_frame(ctx['frame']),ctx['request'])
        ctx['plan']=result;return result

class PlannerAgent(Agent):
    id='planning';title='Planning'
    def run(self,ctx,settings,path,event):
        plan=ctx['plan'];plan['steps']=[agent.title for agent in AGENTS if agent.id not in ('understanding','planning')]
        plan['llm_status']='disabled'
        if ctx['request']['use_llm']:
            try:
                response=LLMClient(settings).reason('Explain the proposed analysis plan. Return {"summary": string, "assumptions": [string]}. Do not change the proposed task or target.',{'goal':ctx['request']['goal'],'task':plan['task'],'target':plan['target'],'columns':list(ctx['frame'].columns),'rows':len(ctx['frame'])})
                if not isinstance(response.get('summary'),str) or not isinstance(response.get('assumptions',[]),list): raise ValueError('Invalid planning response.')
                plan['reasoning']=response['summary'][:4000];plan['assumptions']=[str(v)[:500] for v in response.get('assumptions',[])[:8]];plan['llm_status']='used'
            except Exception as error: plan['llm_status']='fallback';plan['warnings'].append(str(error)[:250]);event(self.id,'LLM unavailable; continuing with the deterministic plan.')
        return plan

class DataProfilingAgent(Agent):
    id='profiling';title='Data Profiling'
    def run(self,ctx,settings,path,event): return describe(ctx['frame'])

class CleaningAgent(Agent):
    id='cleaning';title='Data Cleaning'
    def run(self,ctx,settings,path,event):
        before=len(ctx['frame']);frame=clean_frame(ctx['frame']);target=ctx['plan']['target'];missing=0
        if target:
            missing=int(frame[target].isna().sum());frame=frame.dropna(subset=[target])
            if ctx['plan']['task']=='regression':
                converted=pd.to_numeric(frame[target],errors='coerce')
                if converted.isna().any(): raise ValueError('Regression requires a numeric target.')
                frame[target]=converted
        validate_target(frame,ctx['plan']['task'],target)
        ctx['frame']=frame;export_csv(frame,path/'cleaned.csv')
        return {'original_rows':before,'clean_rows':len(frame),'removed_rows':before-len(frame),'missing_target_rows':missing,'actions':['Trimmed strings and normalized empty/non-finite values.','Removed exact duplicate rows and rows missing the target.','Feature imputation, scaling, missing indicators, and category encoding will fit inside cross-validation.'],'preview':json_safe(frame.head(10).astype(object).where(frame.head(10).notna(),None).to_dict('records'))}

class EDAAgent(Agent):
    id='eda';title='EDA'
    def run(self,ctx,settings,path,event): return eda(ctx['frame'],ctx['plan']['target'])

class FeatureEngineeringAgent(Agent):
    id='features';title='Feature Engineering'
    def run(self,ctx,settings,path,event):
        frame=ctx['frame'];target=ctx['plan']['target'];task=ctx['plan']['task'];seed=ctx['request']['seed']
        train,test,split_warnings=split_supervised(frame,task,target,seed)
        features,excluded,flags=choose_features(train,target,ctx['request'].get('features'),settings.max_features)
        ctx['features']=features;ctx['feature_warnings']=ctx['plan']['warnings']+flags+split_warnings
        if task=='classification':
            folds=min(ctx['request']['cv_folds'],int(train[target].astype(str).value_counts().min()))
            if folds<ctx['request']['cv_folds']:ctx['feature_warnings'].append(f'Cross-validation reduced to {folds} folds to retain every class in each training fold.')
        ctx['X_train']=train[features];ctx['X_test']=test[features]
        if task=='classification':
            encoder=LabelEncoder().fit(train[target].astype(str));ctx['classes']=list(encoder.classes_)
            ctx['y_train']=encoder.transform(train[target].astype(str));ctx['y_test']=encoder.transform(test[target].astype(str))
        elif task=='regression':ctx['y_train']=train[target].to_numpy();ctx['y_test']=test[target].to_numpy()
        else:ctx['y_train']=None;ctx['y_test']=None
        shared=set(pd.util.hash_pandas_object(train[features],index=False))&set(pd.util.hash_pandas_object(test[features],index=False))
        if shared:ctx['feature_warnings'].append('Repeated predictor vectors appear in training and test data; review whether these are duplicate entities requiring grouped validation.')
        return {'selected':features,'excluded':excluded,'train_rows':len(train),'test_rows':len(test),'warnings':ctx['feature_warnings'],'transformations':['Training-fold median imputation with missing-value indicators','Standard scaling for numeric inputs','One-hot categories with bounded cardinality and unknown-category handling']}

class ModelSelectionAgent(Agent):
    id='selection';title='Model Selection'
    def run(self,ctx,settings,path,event):
        from app.ml.training import candidate_specs
        ctx['candidates']=candidate_specs(ctx['plan']['task'],len(ctx['X_train']),ctx['request']['include_dl'])
        return {'candidates':ctx['candidates'],'reason':'Use single-threaded estimators. Larger datasets use Extra Trees or MiniBatch KMeans; neural networks are limited to 10,000 training rows.','baseline':'One-cluster centroid' if ctx['plan']['task']=='clustering' else 'Training mean / majority-class prior','tuning_iterations':ctx['request']['tuning_iterations']}

class TrainingAgent(Agent):
    id='training';title='Model Training & Tuning'
    def run(self,ctx,settings,path,event):return train_models(ctx,settings,path,event)

class EvaluationCriticAgent(Agent):
    id='evaluation';title='Evaluation / Critic'
    def run(self,ctx,settings,path,event):
        result=evaluate(ctx);atomic_joblib(saved_bundle(ctx),path/'model.joblib')
        predictions=pd.DataFrame(result['predictions'])
        export_csv(predictions,path/'predictions.csv')
        return result

class SHAPExplainabilityAgent(Agent):
    id='shap';title='SHAP Explainability'
    def run(self,ctx,settings,path,event):return explain(ctx,settings)

class BusinessInsightAgent(Agent):
    id='insights';title='Business Insights'
    def run(self,ctx,settings,path,event):
        evaluation=ctx['evaluation'];metrics=evaluation['metrics'];task=ctx['plan']['task'];top=[r['feature'] for r in ctx['outputs']['shap'].get('global',[])[:3]]
        summary=f"Selected {ctx['winner']} using cross-validation. Evaluated on {evaluation['test_rows']} held-out rows. "
        summary+=f"Accuracy {metrics['accuracy']:.1%}; macro F1 {metrics['macro_f1']:.3f}." if task=='classification' else f"RMSE {metrics['rmse']:.3f}; R² {metrics['r2']:.3f}." if task=='regression' else f"Observed {metrics['clusters_observed']} clusters; silhouette {metrics['silhouette']}."
        recommendations=['Validate on a fresh independent dataset before operational use.','Review incorrect predictions and missing/underrepresented examples.']
        if top: recommendations.append('Investigate associations involving '+', '.join(top)+'. Confirm possible interventions with a controlled experiment; SHAP does not establish causality.')
        if task=='classification':recommendations.append('Choose a decision threshold using the business costs of false positives and false negatives, then validate it on separate data.')
        elif task=='clustering':recommendations.append('Profile each segment and test whether it supports a useful business action; cluster IDs have no ordinal meaning.')
        if 'churn' in ctx['request']['goal'].lower():recommendations.append('Test targeted retention outreach with a control group and track incremental retained customers before estimating return on investment.')
        result={'summary':summary,'recommendations':recommendations,'source':'measured results and deterministic guidance','llm_status':'disabled'}
        if ctx['request']['use_llm']:
            try:
                response=LLMClient(settings).reason('Return {"summary":string,"recommendations":[string]}. Propose cautious business actions grounded in the measurements. Do not invent revenue impact.',{'goal':ctx['request']['goal'],'task':task,'metrics':metrics,'baseline':evaluation['baseline'],'important_features':top,'warnings':evaluation['warnings']})
                if not isinstance(response.get('summary'),str) or not isinstance(response.get('recommendations'),list):raise ValueError('Invalid insight response.')
                result['llm_commentary']=response['summary'][:4000];result['llm_recommendations']=[str(v)[:600] for v in response['recommendations'][:8]];result['llm_status']='used'
            except Exception as error:result['llm_status']='fallback';result['llm_error']=str(error)[:250]
        return result

class ReportAgent(Agent):
    id='report';title='Report'
    def run(self,ctx,settings,path,event):
        outputs=ctx['outputs'];result={'id':ctx['id'],'plan':ctx['plan'],'profile':outputs['profiling'],'cleaning':outputs['cleaning'],'eda':outputs['eda'],'features':outputs['features'],'models':outputs['training'],'evaluation':outputs['evaluation'],'explainability':outputs['shap'],'insights':outputs['insights'],'model_schema':saved_bundle(ctx)['feature_schema']}
        original=pd.read_csv(settings.data_root/(ctx['request']['dataset_id']+'.csv'))
        result=prepare_report(result,filename=ctx.get('dataset_filename'),request=ctx['request'],frame=original)
        ctx['result']=result
        return build_reports(result,path)

AGENTS=[ProblemUnderstandingAgent(),PlannerAgent(),DataProfilingAgent(),CleaningAgent(),EDAAgent(),FeatureEngineeringAgent(),ModelSelectionAgent(),TrainingAgent(),EvaluationCriticAgent(),SHAPExplainabilityAgent(),BusinessInsightAgent(),ReportAgent()]
