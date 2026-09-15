import json
import numpy as np
import pandas as pd
import pytest
from app.tools.dashboard import dataset_dashboard, dataset_relationship


def test_dashboard_counts_missing_values_categories_and_real_statistics():
    frame=pd.DataFrame({'value':[1.,2.,3.,None,100.], 'category':['a','a','b',None,'b'], 'empty':[None]*5})
    result=dataset_dashboard(frame,'value')
    assert result['rows']==5 and result['column_count']==3
    assert result['missing_cells']==7
    distribution=result['distributions'][0]
    assert sum(b['count'] for b in distribution['bins'])==4
    assert distribution['stats']['median']==2.5
    assert distribution['outliers']==1
    assert result['missing'][0]['column']=='empty'
    assert result['distributions'][2]['bins']==[]
    assert result['correlations']['values']==[[1.]]
    json.dumps(result,allow_nan=False)


def test_category_tail_is_counted_and_large_number_ranges_are_distinct():
    frame=pd.DataFrame({'sensor_id':np.arange(6760000000,6760000100), 'group':[f'group {i}' for i in range(100)]})
    result=dataset_dashboard(frame)
    numeric,categorical=result['distributions']
    assert len({b['label'] for b in numeric['bins']})==len(numeric['bins'])
    assert sum(b['count'] for b in categorical['bins'])==100
    assert categorical['bins'][-1]['other'] is True
    assert result['correlations']['columns']==[]


def test_correlations_keep_undefined_values_and_include_target():
    frame=pd.DataFrame({f'feature_{i}':np.arange(30)+i for i in range(15)})
    frame['id']=range(30);frame['constant']=1;frame['Price']=np.arange(30)*3
    result=dataset_dashboard(frame,'Price')
    assert 'Price' in result['correlations']['columns']
    assert 'id' not in result['correlations']['columns']
    assert len(result['correlations']['columns'])==12
    constant=dataset_dashboard(frame[['constant','Price']])
    assert constant['correlations']['values'][0]==[None,None]
    assert constant['top_correlations']==[]


def test_scatter_sample_is_bounded_reproducible_and_uses_full_data_correlation():
    frame=pd.DataFrame({'x':np.arange(1000,dtype=float),'y':np.arange(1000,dtype=float)*3})
    frame.loc[0,'x']=np.inf
    result=dataset_relationship(frame,'x','y')
    assert result==dataset_relationship(frame,'x','y')
    assert result['valid_rows']==999 and result['omitted_rows']==1
    assert result['sampled'] and result['sample_rows']==600
    assert result['correlation']==pytest.approx(1)
    assert result['bounds']['x']==[1.,999.]
    json.dumps(result,allow_nan=False)
    with pytest.raises(ValueError):dataset_relationship(frame,'x','x')
    with pytest.raises(ValueError):dataset_relationship(frame,'unknown','y')


def test_empty_and_constant_pairs_are_safe():
    result=dataset_relationship(pd.DataFrame({'x':[1.,1.],'y':[2.,2.]}),'x','y')
    assert result['correlation'] is None
    result=dataset_relationship(pd.DataFrame({'x':[np.nan],'y':[np.nan]}),'x','y')
    assert result['points']==[] and result['bounds'] is None
    json.dumps(result,allow_nan=False)


def test_dashboard_api_works_before_training_and_validates_axes(workspace):
    _,_,client=workspace
    dataset=client.post('/api/datasets/sample').json()
    base=f"/api/datasets/{dataset['id']}"
    response=client.get(base+'/eda',params={'target':'churn'})
    assert response.status_code==200
    assert response.json()['filename']=='synthetic-customer-churn.csv'
    assert response.json()['rows']==600
    assert client.get('/api/analyses').json()==[]
    saved=client.get('/api/datasets').json()
    assert saved[0]['id']==dataset['id'] and saved[0]['rows']==600
    scatter=client.get(base+'/relationship',params={'x':'tenure_months','y':'monthly_charge'})
    assert scatter.status_code==200 and scatter.json()['sample_rows']==600
    assert client.get(base+'/eda',params={'target':'invalid'}).status_code==422
    assert client.get(base+'/relationship',params={'x':'churn','y':'monthly_charge'}).status_code==422
    assert client.get(base+'/eda',headers={'X-API-Key':'wrong'}).status_code==401
