import {useEffect,useState} from 'react';
import {BarChart3,CheckCircle2,Database,LoaderCircle,RefreshCw,ScatterChart} from 'lucide-react';
import {api} from './api';
import type {DashboardData,Distribution,Relationship} from './eda-types';

const fmt=(value:number|null|undefined,digits=2)=>value==null?'—':value.toLocaleString(undefined,{maximumFractionDigits:digits});
const tick=(value:number)=>Math.abs(value)>=10000?Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:2}).format(value):fmt(value,3);
const short=(value:string,length=22)=>value.length>length?value.slice(0,length-1)+'…':value;
const colors=['#8faaff','#59d5bf'];

type DatasetOption={id:string;filename:string;rows:number;column_count:number;created:number};
export function EDAWorkspace({apiKey,initialDatasetId,initialTarget}:{apiKey:string;initialDatasetId?:string;initialTarget?:string|null}){
  const [datasets,setDatasets]=useState<DatasetOption[]|null>(null),[selected,setSelected]=useState(''),[error,setError]=useState(''),[attempt,setAttempt]=useState(0);
  useEffect(()=>{
    const controller=new AbortController();setError('');
    void api<DatasetOption[]>('/datasets',apiKey,{signal:controller.signal}).then(rows=>{
      if(controller.signal.aborted)return;setDatasets(rows);setSelected(rows.some(row=>row.id===initialDatasetId)?initialDatasetId!:rows[0]?.id||'');
    }).catch(e=>{if(!controller.signal.aborted)setError(e.message);});
    return()=>controller.abort();
  },[apiKey,initialDatasetId,attempt]);
  if(error)return <section className="surface"><p role="alert">{error}</p><button className="secondary" onClick={()=>setAttempt(value=>value+1)}>Retry datasets</button></section>;
  if(!datasets)return <section className="surface eda-loading" role="status"><LoaderCircle className="spin" size={24}/><p>Loading your datasets…</p></section>;
  if(!datasets.length)return <section className="surface"><ChartEmpty text="Upload a dataset in New Analysis to start exploring. Model training is not required."/></section>;
  return <><div className="eda-dataset-picker"><label>Dataset<select aria-label="EDA dataset" value={selected} onChange={event=>setSelected(event.target.value)}>{datasets.map(item=><option key={item.id} value={item.id}>{item.filename} · {fmt(item.rows)} rows · {new Date(item.created*1000).toLocaleString()}</option>)}</select></label><span className="muted small">Explore any upload, without running a model.</span></div>{selected&&<EDADashboard key={selected} datasetId={selected} target={selected===initialDatasetId?initialTarget||null:null} apiKey={apiKey}/>}</>;
}

export default function EDADashboard({datasetId,target,apiKey}:{datasetId:string;target:string|null;apiKey:string}){
  const [data,setData]=useState<DashboardData|null>(null),[error,setError]=useState(''),[refresh,setRefresh]=useState(0);
  const [column,setColumn]=useState(''),[x,setX]=useState(''),[y,setY]=useState('');
  const [relationship,setRelationship]=useState<Relationship|null>(null),[scatterError,setScatterError]=useState('');
  useEffect(()=>{
    const controller=new AbortController();setData(null);setError('');setRelationship(null);
    const query=target?`?target=${encodeURIComponent(target)}`:'';
    void api<DashboardData>(`/datasets/${datasetId}/eda${query}`,apiKey,{signal:controller.signal}).then(value=>{
      if(controller.signal.aborted)return;setData(value);
      setColumn(value.target||value.relationship_columns[0]||value.distributions[0]?.column||'');
      const preferredY=value.target&&value.relationship_columns.includes(value.target)?value.target:value.relationship_columns[1]||'';
      setY(preferredY);setX(value.relationship_columns.find(name=>name!==preferredY)||'');
    }).catch(e=>{if(!controller.signal.aborted)setError(e.message);});
    return()=>controller.abort();
  },[datasetId,target,apiKey,refresh]);
  useEffect(()=>{
    const controller=new AbortController();setRelationship(null);setScatterError('');
    if(!data||!x||!y||x===y)return;
    void api<Relationship>(`/datasets/${datasetId}/relationship?${new URLSearchParams({x,y})}`,apiKey,{signal:controller.signal}).then(value=>{if(!controller.signal.aborted)setRelationship(value);}).catch(e=>{if(!controller.signal.aborted)setScatterError(e.message);});
    return()=>controller.abort();
  },[data,datasetId,x,y,apiKey]);
  if(error)return <section className="surface"><p role="alert" className="error-text">{error}</p><button className="secondary" onClick={()=>setRefresh(v=>v+1)}><RefreshCw size={16}/>Retry dashboard</button></section>;
  if(!data)return <section className="surface eda-loading" role="status"><LoaderCircle className="spin" size={25}/><h2>Building your data dashboard</h2><p>Computing distributions and relationships from the saved dataset.</p></section>;
  const distribution=data.distributions.find(item=>item.column===column)||data.distributions[0];
  const missing=data.missing.filter(item=>item.count>0);
  return <div className="eda-dashboard">
    <section className="eda-banner"><div className="eda-dataset-icon"><Database size={25}/></div><div><span className="eyebrow">EXPLORE YOUR DATA</span><h2>{data.filename}</h2><p>Original uploaded data · {fmt(data.rows)} records · {data.target?`Target: ${data.target}`:'No target required'}</p></div><span className="eda-source"><span/>Live dataset</span></section>
    <div className="stats-grid eda-stats">{[['Records',fmt(data.rows),'Rows available to explore'],['Features',fmt(data.column_count),`${data.types[0].count} numeric · ${data.types[1].count} categorical`],['Completeness',`${fmt(data.completeness)}%`,`${fmt(data.missing_cells)} missing cells`],['Duplicate rows',fmt(data.duplicate_rows),'Exact copies in the upload']].map(([label,value,detail])=><div key={label}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>)}</div>
    <div className="eda-grid">
      <section className="surface eda-distribution"><div className="section-title"><BarChart3 size={20}/><h2>Distribution explorer</h2></div><label className="eda-selector">Explore a column<select aria-label="Distribution column" value={distribution?.column||''} onChange={e=>setColumn(e.target.value)}>{data.distributions.map(item=><option key={item.column} value={item.column}>{item.column}</option>)}</select></label>{distribution&&<><div className="eda-chart-meta"><span className="pill">{distribution.kind}</span><span>{fmt(distribution.unique)} unique · {fmt(distribution.missing)} missing</span></div><DistributionChart distribution={distribution}/><div className="eda-mini-stats">{(distribution.kind==='numeric'?[['Mean',fmt(distribution.stats.mean)],['Median',fmt(distribution.stats.median)],['Std. deviation',fmt(distribution.stats.std)],['Potential outliers',fmt(distribution.outliers)]]:[['Observed values',fmt(distribution.count)],['Categories',fmt(distribution.unique)]]).map(([label,value])=><div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div><p className="muted small">{distribution.kind==='numeric'?'Outliers use the 1.5 × IQR rule. Hover over a bar for its exact interval.':'Showing the ten most frequent categories; remaining categories are combined.'}</p><details className="eda-details"><summary>View chart data</summary><div className="table-scroll"><table><thead><tr><th>{distribution.kind==='numeric'?'Interval':'Category'}</th><th>Records</th></tr></thead><tbody>{distribution.bins.map((bin,i)=><tr key={i}><td>{bin.label}</td><td>{fmt(bin.count)}</td></tr>)}</tbody></table></div></details></>}</section>
      <section className="surface"><div className="section-title"><h2>Data composition</h2><span className="pill push">{data.column_count} columns</span></div><TypeDonut data={data}/><div className="eda-type-legend">{data.types.map((item,i)=><div key={item.label}><i style={{background:colors[i]}}/><span>{item.label}</span><strong>{item.count}</strong></div>)}</div><div className="eda-quality"><CheckCircle2 size={20}/><div><strong>{data.completeness===100?'All cells are populated':`${fmt(data.completeness)}% of cells are populated`}</strong><p>Explore missing values and unusual ranges before choosing your features.</p></div></div></section>
      <section className="surface"><div className="section-title"><ScatterChart size={20}/><h2>Explore relationships</h2></div>{data.relationship_columns.length<2?<ChartEmpty text="Add at least two numeric, non-identifier columns to explore a scatter plot."/>:<><div className="eda-axis-selectors"><label>X axis<select aria-label="Scatter X axis" value={x} onChange={e=>{const next=e.target.value;setX(next);if(next===y)setY(x);}}>{data.relationship_columns.map(name=><option key={name}>{name}</option>)}</select></label><label>Y axis<select aria-label="Scatter Y axis" value={y} onChange={e=>{const next=e.target.value;setY(next);if(next===x)setX(y);}}>{data.relationship_columns.map(name=><option key={name}>{name}</option>)}</select></label></div>{scatterError?<p className="error-text" role="alert">{scatterError}</p>:!relationship?<div className="eda-chart-loading" role="status"><LoaderCircle className="spin" size={20}/>Loading relationship…</div>:<><ScatterPlot data={relationship}/><div className="eda-scatter-caption"><span className="pill">Pearson r: {fmt(relationship.correlation,3)}</span><span>{fmt(relationship.valid_rows)} valid pairs · {fmt(relationship.omitted_rows)} missing pairs excluded</span></div><p className="muted small">{relationship.sampled?`Showing a reproducible sample of ${fmt(relationship.sample_rows)} points. Correlation and axis ranges use all valid pairs.`:`Showing all ${fmt(relationship.sample_rows)} valid pairs.`} Hover over a point for its source row and values.</p></>}</>}</section>
      <section className="surface"><div className="section-title"><h2>Missing values</h2><span className="pill push">{missing.length} affected columns</span></div>{missing.length?<><div className="eda-missing-bars">{missing.slice(0,10).map(item=><div key={item.column}><div><span title={item.column}>{item.column}</span><strong>{fmt(item.percent)}%</strong></div><div className="eda-missing-track"><i style={{width:`${item.percent}%`}}/></div><small>{fmt(item.count)} of {fmt(data.rows)} records</small></div>)}</div>{missing.length>10&&<p className="muted small">Showing the ten columns with the most missing values.</p>}</>:<div className="eda-complete"><CheckCircle2 size={42}/><h3>No missing values</h3><p>Every cell in this uploaded dataset is populated.</p><span>{fmt(data.rows*data.column_count)} / {fmt(data.rows*data.column_count)} cells</span></div>}<p className="muted small">Missingness describes the original upload, before imputation or training cleanup.</p></section>
    </div>
    <section className="surface"><div className="section-title"><h2>Correlation map</h2><span className="pill push">Pearson correlation</span></div><p className="muted small">Up to 12 numeric columns, including the target when numeric. Identifier columns are excluded. Gray cells mean correlation is undefined.</p>{data.correlations.columns.length<2?<ChartEmpty text="At least two numeric, non-identifier columns are needed for a correlation map."/>:<div className="eda-correlation-layout"><CorrelationMap data={data}/><div className="eda-associations"><h3>Strongest associations</h3><p className="muted small">By absolute correlation in the displayed map.</p>{data.top_correlations.length?data.top_correlations.map(pair=><button key={JSON.stringify([pair.left,pair.right])} className="eda-pair" onClick={()=>{setX(pair.left);setY(pair.right);document.getElementById('eda-scatter-chart')?.scrollIntoView({behavior:'smooth',block:'center'});}} title="Explore this pair in the scatter plot"><span>{pair.left}<small>↔ {pair.right}</small></span><strong>{fmt(pair.value,3)}</strong></button>):<p className="muted">No defined correlations.</p>}</div></div>}<details className="eda-details"><summary>View full correlation values</summary><div className="table-scroll"><table><thead><tr><th>Column</th>{data.correlations.columns.map(c=><th key={c}>{c}</th>)}</tr></thead><tbody>{data.correlations.values.map((row,i)=><tr key={i}><th>{data.correlations.columns[i]}</th>{row.map((value,j)=><td key={j}>{fmt(value,3)}</td>)}</tr>)}</tbody></table></div></details></section>
    <p className="muted small eda-note">{data.note}</p>
  </div>;
}

function ChartEmpty({text}:{text:string}){return <div className="eda-chart-empty"><BarChart3 size={28}/><p>{text}</p></div>;}

function DistributionChart({distribution:d}:{distribution:Distribution}){
  if(!d.bins.length)return <ChartEmpty text="This column has no observed values to chart."/>;
  if(d.kind==='categorical')return <div className="eda-category-bars" role="img" aria-label={`Category counts for ${d.column}`}>{d.bins.map((bin,i)=><div key={i} title={`${bin.label}: ${fmt(bin.count)} records`}><span>{bin.label}</span><div><i style={{width:`${bin.count/Math.max(...d.bins.map(b=>b.count))*100}%`}}/></div><strong>{fmt(bin.count)}</strong></div>)}</div>;
  const w=640,h=258,left=57,right=20,top=22,bottom=45,plotW=w-left-right,plotH=h-top-bottom,max=Math.max(...d.bins.map(bin=>bin.count),1),step=plotW/d.bins.length;
  return <svg className="eda-chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Histogram of ${d.column}; ${d.count} observed records`}><text x={left} y={12} className="eda-axis-text">Records</text>{[0,.25,.5,.75,1].map(part=><g key={part}><line x1={left} x2={w-right} y1={top+plotH*(1-part)} y2={top+plotH*(1-part)} stroke="#2b3b56" strokeDasharray="3 5"/><text x={left-10} y={top+plotH*(1-part)+4} textAnchor="end" className="eda-axis-text">{tick(max*part)}</text></g>)}{d.bins.map((bin,i)=><g key={i}><rect x={left+i*step+2} y={top+plotH*(1-bin.count/max)} width={Math.max(1,step-4)} height={plotH*bin.count/max} fill="#8faaff" rx={3} className="eda-hist-bar"><title>{bin.label}: {fmt(bin.count)} records</title></rect>{i%Math.max(1,Math.ceil(d.bins.length/5))===0&&<text x={left+i*step} y={h-23} className="eda-axis-text" textAnchor="start">{tick(bin.low!)}</text>}</g>)}<text x={w-right} y={h-23} textAnchor="end" className="eda-axis-text">{tick(d.bins.at(-1)!.high!)}</text><text x={left+plotW/2} y={h-3} textAnchor="middle" className="eda-axis-text">{short(d.column,45)}</text></svg>;
}

function TypeDonut({data}:{data:DashboardData}){
  const ratio=data.types[0].count/Math.max(data.column_count,1),radius=68,circumference=2*Math.PI*radius;
  return <svg viewBox="0 0 240 215" className="eda-donut" role="img" aria-label={`${data.types[0].count} numeric columns and ${data.types[1].count} categorical columns`}><circle cx={120} cy={106} r={radius} fill="none" stroke={colors[1]} strokeWidth={21}/><circle cx={120} cy={106} r={radius} fill="none" stroke={colors[0]} strokeWidth={21} strokeDasharray={`${circumference*ratio} ${circumference}`} transform="rotate(-90 120 106)"/><text x={120} y={108} textAnchor="middle" fill="#eff4ff" fontSize={34} fontWeight={650}>{data.column_count}</text><text x={120} y={131} textAnchor="middle" className="eda-axis-text">total columns</text></svg>;
}

function ScatterPlot({data}:{data:Relationship}){
  if(!data.bounds||!data.points.length)return <ChartEmpty text="There are no rows with values in both selected columns."/>;
  const w=640,h=300,left=64,right=18,top=22,bottom=50;
  const domain=(bounds:[number,number])=>{const pad=(bounds[1]-bounds[0])*.04||Math.abs(bounds[0])*.04||1;return [bounds[0]-pad,bounds[1]+pad];};
  const dx=domain(data.bounds.x),dy=domain(data.bounds.y),sx=(v:number)=>left+(v-dx[0])/(dx[1]-dx[0])*(w-left-right),sy=(v:number)=>h-bottom-(v-dy[0])/(dy[1]-dy[0])*(h-top-bottom);
  return <svg id="eda-scatter-chart" className="eda-chart eda-scatter" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Scatter plot of ${data.x} against ${data.y}, ${data.sample_rows} points`}><text x={left} y={12} className="eda-axis-text">{short(data.y,50)}</text>{[0,.25,.5,.75,1].map(part=><g key={part}><line x1={left} x2={w-right} y1={top+(h-top-bottom)*part} y2={top+(h-top-bottom)*part} stroke="#2b3b56" strokeDasharray="3 5"/><text x={left-10} y={top+(h-top-bottom)*part+4} textAnchor="end" className="eda-axis-text">{tick(dy[1]-(dy[1]-dy[0])*part)}</text><text x={left+(w-left-right)*part} y={h-27} textAnchor="middle" className="eda-axis-text">{tick(dx[0]+(dx[1]-dx[0])*part)}</text></g>)}{data.points.map(point=><circle key={point.row} cx={sx(point.x)} cy={sy(point.y)} r={3.4} fill="#6bdbc5" fillOpacity={.6} className="eda-scatter-point"><title>{`Data row ${point.row} · ${data.x}: ${fmt(point.x,6)} · ${data.y}: ${fmt(point.y,6)}`}</title></circle>)}<text x={(left+w-right)/2} y={h-5} textAnchor="middle" className="eda-axis-text">{short(data.x,50)}</text></svg>;
}

function CorrelationMap({data}:{data:DashboardData}){
  const cols=data.correlations.columns,n=cols.length,cell=38,left=156,top=26,w=left+n*cell+10,h=top+n*cell+37;
  return <div className="eda-heatmap"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Numeric correlation heatmap"><text x={left} y={12} className="eda-axis-text">Column number</text>{cols.map((column,i)=><g key={column}><text x={left-10} y={top+i*cell+24} textAnchor="end" className="eda-axis-text"><title>{column}</title>{i+1}. {short(column,20)}</text><text x={left+i*cell+cell/2} y={h-16} textAnchor="middle" className="eda-axis-text">{i+1}</text></g>)}{data.correlations.values.flatMap((row,i)=>row.map((value,j)=><g key={`${i}-${j}`}><rect x={left+j*cell+1} y={top+i*cell+1} width={cell-3} height={cell-3} rx={4} fill={value===null?'#26334b':value<0?`rgba(196,142,235,${.12+Math.abs(value)*.8})`:`rgba(125,165,255,${.12+Math.abs(value)*.8})`}><title>{`${cols[i]} ↔ ${cols[j]}: ${value===null?'undefined':value.toFixed(4)}`}</title></rect><text x={left+j*cell+cell/2-1} y={top+i*cell+23} textAnchor="middle" fill={value!==null&&Math.abs(value)>.6?'#0b1427':'#d3dff1'} fontSize={10} pointerEvents="none">{value===null?'—':value.toFixed(1)}</text></g>))}</svg><div className="eda-scale"><span>−1</span><i/><span>+1</span></div></div>;
}
