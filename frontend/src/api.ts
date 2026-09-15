import {normalizeApiBase} from './api-config';
import {BackendUnavailableError, requestJson} from './transport';

export const BACKEND_UNAVAILABLE = 'autods-backend-unavailable';

export const configuredApiBase=()=>sessionStorage.getItem('autods-server-url')||import.meta.env.VITE_API_BASE_URL||'';
export const apiUrl=(path:string)=>`${normalizeApiBase(configuredApiBase(),window.location.protocol)}${path}`;

export async function api<T>(path:string,key:string,options:RequestInit={}):Promise<T>{
  const headers=new Headers(options.headers);if(key)headers.set('X-API-Key',key);if(options.body&&!(options.body instanceof FormData))headers.set('Content-Type','application/json');
  const url=apiUrl(path);
  try{return await requestJson<T>(url,{...options,headers});}
  catch(error){if(error instanceof BackendUnavailableError)window.dispatchEvent(new Event(BACKEND_UNAVAILABLE));throw error;}
}
export async function downloadArtifact(id:string,kind:string,key:string){
  const response=await fetch(apiUrl(`/analyses/${id}/artifacts/${kind}`),{headers:key?{'X-API-Key':key}:{}});
  if(!response.ok)throw Error('Artifact is unavailable. Complete the analysis first.');
  const blob=await response.blob(),url=URL.createObjectURL(blob),anchor=document.createElement('a');
  anchor.href=url;anchor.download=`analysis-${id}.${({model:'joblib',predictions:'csv',cleaned:'csv'} as Record<string,string>)[kind]||kind}`;
  document.body.appendChild(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);
}
