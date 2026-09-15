import time
import traceback
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from app.database.store import Store
from app.settings import Settings
from app.agents.stages import AGENTS
from app.ml.training import atomic_joblib
from app.tools.data import json_safe

class OrchestratorAgent:
    def __init__(self,settings):self.settings=settings;self.store=Store(settings.db_path)

    def run(self,analysis_id):
        record=self.store.analysis(analysis_id);path=self.settings.project_root/analysis_id;path.mkdir(exist_ok=True)
        state=path/'checkpoint.joblib'
        event=lambda stage,message:self.store.event(analysis_id,stage,message)
        try:
            if state.exists():
                ctx=joblib.load(state)
                if ctx['fingerprint']!=record['fingerprint']:raise ValueError('Checkpoint configuration does not match this analysis.')
            else:
                frame=pd.read_csv(self.settings.data_root/(record['dataset_id']+'.csv'))
                ctx={'id':analysis_id,'request':record['request'],'fingerprint':record['fingerprint'],'frame':frame,'outputs':{}}
            ctx['dataset_filename']=record['dataset_filename']
            with threadpool_limits(limits=1):
                for agent in AGENTS:
                    if self.store.analysis(analysis_id)['cancel_requested']:raise InterruptedError('Cancelled by user.')
                    if agent.id in ctx['outputs']:
                        self.store.checkpoint(analysis_id,agent.id,ctx['outputs'][agent.id],0);event(agent.id,'Using cached agent checkpoint.');continue
                    self.store.update(analysis_id,stage=agent.id)
                    event(agent.id,agent.title+' started.');started=time.monotonic()
                    for attempt in range(2):
                        try:output=agent.run(ctx,self.settings,path,event);break
                        except (TimeoutError,ConnectionError) as error:
                            if attempt:raise
                            event(agent.id,f'Transient error; retrying once: {str(error)[:200]}');time.sleep(.5)
                    ctx['outputs'][agent.id]=json_safe(output)
                    atomic_joblib(ctx,state)
                    self.store.checkpoint(analysis_id,agent.id,ctx['outputs'][agent.id],time.monotonic()-started)
                    event(agent.id,agent.title+' completed.')
            self.store.update(analysis_id,status='completed',stage='report',result=ctx['result'],error=None)
            event('completed','Analysis completed. Reports and saved model are ready.')
        except InterruptedError as error:self.store.update(analysis_id,status='cancelled',error=str(error));event('cancelled',str(error))
        except Exception as error:
            # Detailed tracebacks remain in the private project directory, not API responses.
            (path/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
            self.store.update(analysis_id,status='failed',error=str(error)[:1000]);event('failed',str(error)[:500])

def run_child(analysis_id,settings_dict):
    OrchestratorAgent(Settings(**settings_dict)).run(analysis_id)
