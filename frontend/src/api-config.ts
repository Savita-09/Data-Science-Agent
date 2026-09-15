/** Resolve a backend address without allowing credentials or mixed-content requests. */
export function normalizeApiBase(value:string,pageProtocol='http:'):string{
  const address=value.trim();
  if(!address)return '/api';
  let url:URL;
  try{url=new URL(address);}catch{throw Error('Enter a full backend URL, such as https://api.example.com.');}
  if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash){
    throw Error('Use an HTTP or HTTPS backend URL without credentials, a query, or a fragment.');
  }
  if(pageProtocol==='https:'&&url.protocol!=='https:')throw Error('This secure website requires an HTTPS backend URL.');
  const path=url.pathname.replace(/\/+$/,'');
  return `${url.origin}${path.endsWith('/api')?path:`${path}/api`}`;
}
