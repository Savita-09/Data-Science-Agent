import copy
import io
import numpy as np
import pandas as pd
from pypdf import PdfReader
from app.reporting.reports import prepare_report, render_html, render_pdf, report_outline
from app.tools.data import describe


def report_fixture():
    frame=pd.DataFrame({'amount':np.arange(36,dtype=float),'visits':np.arange(36,dtype=float)*2,'segment':['A','B','C']*12,'outcome':['Yes','No']*18})
    frame.loc[0,'amount']=np.nan
    result={
        'id':'synthetic-report-fixture','plan':{'summary':'Predict outcome for a synthetic dataset','task':'classification','target':'outcome'},
        'profile':describe(frame),'cleaning':{'original_rows':36,'clean_rows':36,'removed_rows':0,'actions':['Impute numeric features in training folds.']},
        'eda':{'distributions':[],'correlations':{'columns':[],'values':[]},'note':'Legacy EDA'},
        'features':{'selected':['amount','visits','segment'],'excluded':[],'transformations':['Fit preprocessing inside cross-validation.']},
        'models':{'winner':'linear','selection_metric':'macro_f1','cv_folds':3,'leaderboard':[{'name':'linear','family':'ML','cv_score':.8,'cv_std':.02,'seconds':1,'status':'completed','parameters':{'C':1}}]},
        'evaluation':{'metrics':{'accuracy':.8,'confusion_matrix':[[3,1],[1,5]]},'baseline':{'accuracy':.6},'training_metrics':{'accuracy':.85},'classes':['No','Yes'],'train_rows':26,'test_rows':10,'warnings':['Synthetic model metrics; not a real training result.'],'predictions':[]},
        'explainability':{'status':'completed','global':[{'feature':'amount','mean_abs_shap':.2}],'local':[],'sample_rows':3,'method':'permutation','units':'probability','note':'Associations are not causes.'},
        'insights':{'summary':'Synthetic report fixture.','recommendations':['Validate independently.']},
        'model_schema':[{'name':'amount','type':'numeric'}],
    }
    return frame,result


def test_main_report_contains_all_stages_and_measured_eda():
    frame,result=report_fixture()
    original=copy.deepcopy(result)
    report=prepare_report(result,filename='synthetic-marketing.csv',request={'seed':42,'tuning_iterations':2},frame=frame)
    assert result==original  # Preparation cannot rewrite saved evidence.
    assert report['report_eda']['rows']==36
    assert len(report['report_eda']['relationship']['points'])==35
    for distribution in report['report_eda']['distributions']:
        assert sum(v['count'] for v in distribution['bins'])==frame[distribution['column']].count()
    titles=[s['title'] for s in report_outline(report)]
    for title in ('1. Problem and plan','3. Dataset profile and quality','4. Exploratory data analysis (EDA)','Numeric correlations','Numeric relationship','5. Cleaning and feature engineering','6. Cross-validation leaderboard','7. Held-out evaluation','8. SHAP explainability','9. Recommendations and limitations','10. Using the saved model'):
        assert title in titles
    document=render_html(report)
    assert 'synthetic-marketing.csv' in document and document.count('<svg')>=7
    assert 'Dataset preview (first 5 rows)' in document
    assert '<script' not in document and 'https://' not in document
    pdf=PdfReader(io.BytesIO(render_pdf(report)))
    text='\n'.join(page.extract_text() for page in pdf.pages)
    assert len(pdf.pages)>=4
    for expected in ('synthetic-marketing.csv','Exploratory data analysis','Distribution: amount','Numeric correlations','Cross-validation leaderboard','Confusion matrix','SHAP explainability','Using the saved model'):
        assert expected in text


def test_report_escapes_untrusted_labels_and_handles_missing_correlations():
    frame,result=report_fixture()
    label='<script>alert(1)</script>'
    frame['segment']=[label]*len(frame)
    frame['visits']=None
    result['profile']=describe(frame)
    result['plan']['summary']=label
    report=prepare_report(result,filename=label,frame=frame)
    document=render_html(report)
    assert label not in document and '&lt;script&gt;' in document
    assert '<script' not in document.lower()
    pdf=PdfReader(io.BytesIO(render_pdf(report)))
    assert len(pdf.pages)>1


def test_saved_analysis_downloads_get_main_report_without_retraining(workspace,monkeypatch):
    settings,store,client=workspace
    frame,result=report_fixture()
    uploaded=client.post('/api/datasets',files={'file':('existing.csv',frame.to_csv(index=False).encode(),'text/csv')}).json()
    submitted=client.post('/api/analyses',json={'dataset_id':uploaded['id'],'goal':'Predict outcome','target':'outcome','time_budget_seconds':120})
    assert submitted.status_code==202,submitted.text
    job=submitted.json()['id'];result['id']=job
    store.update(job,status='completed',result=result)
    before=store.analysis(job)
    def no_training(*args,**kwargs):raise AssertionError('Downloading a report must not retrain')
    monkeypatch.setattr('app.agents.stages.TrainingAgent.run',no_training)
    for kind in ('html','pdf','json'):
        response=client.get(f'/api/analyses/{job}/artifacts/{kind}')
        assert response.status_code==200,response.text[:200]
        assert 'attachment' in response.headers['content-disposition']
        if kind=='html':assert 'Exploratory data analysis' in response.text and 'existing.csv' in response.text
        if kind=='pdf':assert 'existing.csv' in PdfReader(io.BytesIO(response.content)).pages[0].extract_text()
        if kind=='json':assert response.json()['report_eda']['rows']==36
    after=store.analysis(job)
    assert after['result']==before['result'] and after['updated']==before['updated']
    assert not (settings.project_root/job/'model.joblib').exists()
    assert client.get(f'/api/analyses/{job}/artifacts/unknown').status_code==404
