import {normalizeApiBase} from './api-config';

export const configuredApiBase=()=>sessionStorage.getItem('autods-server-url')||import.meta.env.VITE_API_BASE_URL||'';
export const apiUrl=(path:string)=>`${normalizeApiBase(configuredApiBase(),window.location.protocol)}${path}`;

export async function api<T>(path:string,key:string,options:RequestInit={}):Promise<T>{
  const headers=new Headers(options.headers);if(key)headers.set('X-API-Key',key);if(options.body&&!(options.body instanceof FormData))headers.set('Content-Type','application/json');
  let response:Response;
  const url=apiUrl(path);
  try{response=await fetch(url,{...options,headers});}catch{throw Error('Cannot reach the Python backend. Check its URL, start the API and worker, and allow this website in the backend CORS settings.');}
  if(!response.ok){let detail='Request failed';try{const body=await response.json();detail=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail);}catch{detail=response.statusText;}throw Error(`${response.status}: ${detail}`);}
  if(!response.headers.get('content-type')?.includes('application/json'))throw Error('Python backend is not configured for this website. Open API connection and enter your backend URL.');
  return response.json();
}
export async function downloadArtifact(id:string,kind:string,key:string){
  const response=await fetch(apiUrl(`/analyses/${id}/artifacts/${kind}`),{headers:key?{'X-API-Key':key}:{}});
  if(!response.ok)throw Error('Artifact is unavailable. Complete the analysis first.');
  const blob=await response.blob(),url=URL.createObjectURL(blob),anchor=document.createElement('a');
  anchor.href=url;anchor.download=`analysis-${id}.${({model:'joblib',predictions:'csv',cleaned:'csv'} as Record<string,string>)[kind]||kind}`;
  document.body.appendChild(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);
}
