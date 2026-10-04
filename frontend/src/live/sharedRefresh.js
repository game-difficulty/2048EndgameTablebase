// Coalesce socket-open / foreground refreshes, including a just-completed request.
export function createSharedRefresh(refresh, now = () => Date.now()) {
  let job=null, completed=-Infinity, result;
  return () => {
    if(job)return job;
    if(now()-completed<2000)return Promise.resolve(result);
    job=Promise.resolve().then(refresh).then(value=>{
      if(value!==undefined){result=value;completed=now();}
      return value;
    }).finally(()=>{job=null;});
    return job;
  };
}
