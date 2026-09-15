import os
import time
import warnings
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, root_mean_squared_error, r2_score, silhouette_score, davies_bouldin_score
from xgboost import XGBClassifier, XGBRegressor
from app.preprocessing.pipeline import make_preprocessor
from app.tools.data import json_safe

def atomic_joblib(value, path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary,path)

def cluster_score(estimator, X, y=None):
    values=estimator.named_steps['preprocess'].transform(X)
    labels=estimator.predict(X)
    if not 1<len(np.unique(labels))<len(labels): return -1.0
    return silhouette_score(values,labels,sample_size=min(1000,len(labels)),random_state=42)

def candidate_specs(task, rows, include_dl=True):
    if task=='clustering': return ['kmeans']
    return ['baseline','linear','extra_trees' if rows>10000 else 'random_forest','xgboost']+(['neural_network'] if include_dl and rows<=10000 else [])

def make_candidate(name, task, rows, seed, quick=False):
    classification=task=='classification'
    if name=='baseline': return (DummyClassifier(strategy='prior') if classification else DummyRegressor(strategy='mean')),{}
    if name=='linear': return (LogisticRegression(max_iter=1000,random_state=seed) if classification else Ridge()),({'model__C':[.1,1,10],'model__class_weight':[None,'balanced']} if classification else {'model__alpha':[.1,1,10,100]})
    if name in ('random_forest','extra_trees'):
        cls=(ExtraTreesClassifier if classification else ExtraTreesRegressor) if name=='extra_trees' else (RandomForestClassifier if classification else RandomForestRegressor)
        model=cls(n_estimators=40 if quick else 100,random_state=seed,n_jobs=1)
        return model,{'model__max_depth':[4,8,None],'model__min_samples_leaf':[2,5,10],**({'model__class_weight':[None,'balanced']} if classification else {})}
    if name=='xgboost':
        cls=XGBClassifier if classification else XGBRegressor
        return cls(n_estimators=40 if quick else 120,n_jobs=1,random_state=seed,tree_method='hist',verbosity=0),{'model__max_depth':[2,4,6],'model__learning_rate':[.03,.1],'model__subsample':[.8,1.0],'model__reg_lambda':[1,5]}
    if name=='neural_network':
        cls=MLPClassifier if classification else MLPRegressor
        # A validation partition internal to each fit controls early stopping.
        return cls(hidden_layer_sizes=(32,16),early_stopping=True,validation_fraction=.15,n_iter_no_change=12,max_iter=80 if quick else 200,random_state=seed),{'model__alpha':[.001,.01],'model__learning_rate_init':[.001,.003]}
    if name=='kmeans':
        cls=MiniBatchKMeans if rows>10000 else KMeans
        return cls(random_state=seed,n_init=10),{'model__n_clusters':[2,3,4,5,6]}
    raise ValueError('Unknown model candidate.')

def train_models(ctx, settings, project_dir, event):
    X,y=ctx['X_train'],ctx['y_train'];task=ctx['plan']['task'];request=ctx['request'];seed=request['seed']
    prep=make_preprocessor(X)
    encoded=prep.fit_transform(X)
    if encoded.shape[1]>settings.max_encoded_features: raise ValueError(f'Encoding produces {encoded.shape[1]} features; limit is {settings.max_encoded_features}. Select fewer categorical features.')
    folds=min(request['cv_folds'],int(np.bincount(y).min())) if task=='classification' else request['cv_folds']
    if folds<2: raise ValueError('Too few examples per class for cross-validation.')
    cv=StratifiedKFold(folds,shuffle=True,random_state=seed) if task=='classification' else KFold(folds,shuffle=True,random_state=seed)
    scoring='f1_macro' if task=='classification' else 'neg_root_mean_squared_error' if task=='regression' else cluster_score
    leaderboard=[]; fitted={}; cache_dir=project_dir/'model-cache';cache_dir.mkdir(exist_ok=True)
    for name in ctx['candidates']:
        started=time.monotonic();cache=cache_dir/f'{name}.joblib'
        if cache.exists():
            saved=joblib.load(cache)
            if saved['fingerprint']==ctx['fingerprint']:
                leaderboard.append(saved['summary']);fitted[name]=saved['pipeline'];event('training',f'Reused completed {name} training from checkpoint.');continue
        event('training',f'Training {name} with {folds}-fold cross-validation.')
        model,parameters=make_candidate(name,task,len(X),seed,request.get('quick',False))
        if name=='neural_network' and task=='classification':
            smallest_inner=int(np.bincount(y).min())*(folds-1)//folds
            if smallest_inner<2 or int(len(X)*(folds-1)/folds*.15)<len(np.unique(y)):
                model.set_params(early_stopping=False)
                event('training','Small classes: neural network uses bounded epochs and cross-validation without an extra early-stopping split.')
        pipeline=Pipeline([('preprocess',clone(prep)),('model',model)])
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if parameters:
                    search=RandomizedSearchCV(pipeline,parameters,n_iter=request['tuning_iterations'],scoring=scoring,cv=cv,n_jobs=1,refit=True,random_state=seed,error_score='raise',return_train_score=True)
                    search.fit(X,y);pipeline=search.best_estimator_;index=search.best_index_
                    cv_mean=float(search.best_score_);cv_std=float(search.cv_results_['std_test_score'][index]);params=search.best_params_
                else:
                    scores=cross_val_score(pipeline,X,y,scoring=scoring,cv=cv,n_jobs=1,error_score='raise')
                    cv_mean=float(scores.mean());cv_std=float(scores.std());params={};pipeline.fit(X,y)
            if not np.isfinite(cv_mean): raise ValueError('Non-finite validation score.')
            summary={'name':name,'family':'DL' if name=='neural_network' else 'Baseline' if name=='baseline' else 'ML','status':'completed','cv_score':cv_mean,'cv_std':cv_std,'parameters':json_safe(params),'seconds':round(time.monotonic()-started,2),'warnings':list(dict.fromkeys(str(w.message)[:250] for w in caught))[:3]}
            leaderboard.append(summary);fitted[name]=pipeline
            atomic_joblib({'fingerprint':ctx['fingerprint'],'summary':summary,'pipeline':pipeline},cache)
            event('training',f'{name} completed; validation score {cv_mean:.4f}.')
        except Exception as error:
            leaderboard.append({'name':name,'family':'DL' if name=='neural_network' else 'ML','status':'failed','error':str(error)[:500],'seconds':round(time.monotonic()-started,2)})
            event('training',f'{name} failed: {str(error)[:300]}')
    complete=[r for r in leaderboard if r['status']=='completed']
    if not complete: raise ValueError('All candidate models failed. Review the training log.')
    complete.sort(key=lambda r:r['cv_score'],reverse=True)
    winner=complete[0]
    ctx['pipeline']=fitted[winner['name']];ctx['baseline_pipeline']=fitted.get('baseline');ctx['winner']=winner['name']
    ctx['leaderboard']=complete+[r for r in leaderboard if r['status']!='completed'];ctx['scoring']='macro F1' if task=='classification' else 'negative RMSE' if task=='regression' else 'silhouette'
    return {'winner':winner['name'],'selection_metric':ctx['scoring'],'cv_folds':folds,'leaderboard':ctx['leaderboard'],'note':'The winner was selected using cross-validation on training data only. Test metrics were not used for selection.'}

def supervised_metrics(pipeline, X, y, task):
    predicted=pipeline.predict(X)
    if task=='classification':
        result={'accuracy':accuracy_score(y,predicted),'balanced_accuracy':balanced_accuracy_score(y,predicted),'macro_f1':f1_score(y,predicted,average='macro',zero_division=0),'confusion_matrix':confusion_matrix(y,predicted).tolist()}
        if hasattr(pipeline,'predict_proba'):
            proba=pipeline.predict_proba(X)
            try: result['roc_auc']=roc_auc_score(y,proba[:,1]) if proba.shape[1]==2 else roc_auc_score(y,proba,multi_class='ovr')
            except ValueError: result['roc_auc']=None
        return json_safe(result),predicted
    return json_safe({'rmse':root_mean_squared_error(y,predicted),'mae':mean_absolute_error(y,predicted),'r2':r2_score(y,predicted)}),predicted

def evaluate(ctx):
    pipeline=ctx['pipeline'];task=ctx['plan']['task'];X=ctx['X_test'];y=ctx['y_test'];flags=list(ctx['feature_warnings'])
    if task=='clustering':
        values=pipeline.named_steps['preprocess'].transform(X);predicted=pipeline.predict(X);unique=np.unique(predicted)
        valid=1<len(unique)<len(X)
        metrics={'silhouette':float(silhouette_score(values,predicted,sample_size=min(1000,len(X)),random_state=42)) if valid else None,'davies_bouldin':float(davies_bouldin_score(values,predicted)) if valid else None,'clusters_observed':len(unique)}
        train_values=pipeline.named_steps['preprocess'].transform(ctx['X_train'])
        metrics['mean_squared_distance']=float(np.mean(np.min(pipeline.named_steps['model'].transform(values),axis=1)**2))
        baseline={'mean_squared_distance':float(np.mean(np.sum((values-train_values.mean(axis=0))**2,axis=1))),'description':'One-cluster training-mean baseline; silhouette is undefined for one cluster.'}
        train_metrics={'silhouette':float(cluster_score(pipeline,ctx['X_train']))}
        flags.append('Clusters are exploratory segments, not ground-truth classes; there is no clustering accuracy metric.')
        predictions=[{'row':int(index)+1,'actual':None,'predicted':int(p)} for index,p in zip(X.index,predicted)]
    else:
        metrics,predicted=supervised_metrics(pipeline,X,y,task)
        baseline,_=supervised_metrics(ctx['baseline_pipeline'],X,y,task) if ctx['baseline_pipeline'] is not None else ({},None)
        train_metrics,_=supervised_metrics(pipeline,ctx['X_train'],ctx['y_train'],task)
        best_cv=ctx['leaderboard'][0]['cv_score']
        gap=train_metrics['macro_f1']-best_cv if task=='classification' else 1-train_metrics['rmse']/max(-best_cv,1e-12)
        if gap>.15 if task=='classification' else gap>.3: flags.append('Possible overfitting: training performance is materially better than cross-validation performance.')
        decode=lambda v:ctx['classes'][int(v)] if task=='classification' else float(v)
        predictions=[{'row':int(index)+1,'actual':decode(a),'predicted':decode(p)} for index,a,p in zip(X.index,y,predicted)]
    if len(X)<100: flags.append('Small test set: performance estimates may vary substantially on new samples.')
    flags.append('Random splitting assumes independent rows. Use a purpose-built chronological or grouped validation design for repeated entities or time-dependent outcomes.')
    result=json_safe({'task':task,'metrics':metrics,'baseline':baseline,'training_metrics':train_metrics,'train_rows':len(ctx['X_train']),'test_rows':len(X),'classes':ctx.get('classes',[]),'warnings':flags,'predictions':predictions})
    ctx['evaluation']=result
    return result

def saved_bundle(ctx):
    return {'version':1,'pipeline':ctx['pipeline'],'task':ctx['plan']['task'],'target':ctx['plan'].get('target'),'classes':ctx.get('classes',[]),'feature_schema':[{'name':c,'type':'numeric' if c in ctx['X_train'].select_dtypes(include='number') else 'categorical','example':json_safe(ctx['X_train'][c].dropna().iloc[0]) if ctx['X_train'][c].notna().any() else None} for c in ctx['features']], 'fingerprint':ctx['fingerprint'],'winner':ctx['winner']}
