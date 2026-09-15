export async function api<T>(path:string,key:string,options:RequestInit={}):Promise<T>{
  const headers=new Headers(options.headers);if(key)headers.set('X-API-Key',key);if(options.body&&!(options.body instanceof FormData))headers.set('Content-Type','application/json');
  let response:Response;
  try{response=await fetch(`/api${path}`,{...options,headers});}catch{throw Error('Cannot reach the Python backend. Start the API and worker, then reconnect.');}
  if(!response.ok){let detail='Request failed';try{const body=await response.json();detail=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail);}catch{detail=response.statusText;}throw Error(`${response.status}: ${detail}`);}
  return response.json();
}
export async function downloadArtifact(id:string,kind:string,key:string){
  const response=await fetch(`/api/analyses/${id}/artifacts/${kind}`,{headers:key?{'X-API-Key':key}:{}});
  if(!response.ok)throw Error('Artifact is unavailable. Complete the analysis first.');
  const blob=await response.blob(),url=URL.createObjectURL(blob),anchor=document.createElement('a');
  anchor.href=url;anchor.download=`analysis-${id}.${({model:'joblib',predictions:'csv',cleaned:'csv'} as Record<string,string>)[kind]||kind}`;
  document.body.appendChild(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);
}
