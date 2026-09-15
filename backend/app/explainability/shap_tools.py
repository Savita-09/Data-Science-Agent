import numpy as np
import shap
from app.tools.data import json_safe

def explain(ctx, settings):
    pipeline=ctx['pipeline'];prep=pipeline.named_steps['preprocess'];model=pipeline.named_steps['model'];task=ctx['plan']['task']
    background=prep.transform(ctx['X_train'].sample(min(30,len(ctx['X_train'])),random_state=ctx['request']['seed']))
    source=ctx['X_test'].iloc[:settings.shap_rows]
    values=prep.transform(source);names=list(prep.get_feature_names_out())
    if ctx['winner']=='baseline':
        return {'status':'not_applicable','reason':'The selected baseline ignores input features. Every feature contribution is zero; there is no input-dependent model to explain.','global':[],'local':[]}
    if task=='classification':
        predict=lambda X:model.predict_proba(X)
        output_names=ctx['classes'];units='predicted probability'
    elif task=='clustering':
        predict=lambda X:-model.transform(X)
        output_names=[f'Cluster {i}' for i in range(model.n_clusters)];units='negative distance to cluster center'
    else:
        predict=lambda X:model.predict(X)
        output_names=[ctx['plan']['target']];units='predicted target value'
    method='Tree SHAP'
    try:
        if ctx['winner'] not in ('random_forest','extra_trees','xgboost'): raise TypeError('Use model-agnostic SHAP.')
        explainer=shap.TreeExplainer(model,data=background,feature_perturbation='interventional',model_output='probability' if task=='classification' else 'raw')
        result=explainer(values,check_additivity=False)
        contributions=np.asarray(result.values);base=np.asarray(result.base_values)
        # Binary XGBoost Tree SHAP exposes the positive-class output only.
        if task=='classification' and contributions.ndim==2:
            contributions=np.stack([-contributions,contributions],axis=2)
            base=np.stack([1-base,base],axis=1)
    except Exception:
        method='Permutation SHAP'
        explainer=shap.Explainer(predict,shap.maskers.Independent(background[:8]),algorithm='permutation',feature_names=names,seed=ctx['request']['seed'])
        result=explainer(values,max_evals=2*len(names)+1,batch_size=256)
        contributions=np.asarray(result.values);base=np.asarray(result.base_values)
    predicted=np.asarray(predict(values))
    if contributions.ndim==2: contributions=contributions[:,:,None]
    if predicted.ndim==1: predicted=predicted[:,None]
    if base.ndim==1: base=base[:,None]
    global_values=np.abs(contributions).mean(axis=(0,2))
    global_summary=sorted([{'feature':name,'mean_abs_shap':float(value)} for name,value in zip(names,global_values)],key=lambda r:r['mean_abs_shap'],reverse=True)
    local=[]
    for row in range(len(source)):
        output=int(np.argmax(predicted[row])) if task!='regression' else 0
        shap_row=contributions[row,:,output];expected=float(base[row,output]);prediction=float(predicted[row,output])
        parts=sorted([{'feature':name,'value':float(value),'contribution':float(contribution)} for name,value,contribution in zip(names,values[row],shap_row)],key=lambda r:abs(r['contribution']),reverse=True)
        local.append({'row':int(source.index[row])+1,'output':output_names[output],'base_value':expected,'prediction':prediction,'sum_contributions':float(shap_row.sum()),'additivity_error':float(abs(expected+shap_row.sum()-prediction)),'contributions':parts[:25],'omitted_contribution':float(sum(p['contribution'] for p in parts[25:]))})
    return json_safe({'status':'completed','method':method,'units':units,'sample_rows':len(source),'background_rows':len(background) if method=='Tree SHAP' else min(8,len(background)),'global':global_summary,'local':local,'note':'Global importance is mean absolute SHAP on a bounded held-out sample. Local values explain the named output. Encoded categories and missing-value indicators are shown separately. Contributions describe model associations, not causal effects.'})
