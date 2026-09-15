"""Durable worker supervisor. Run separately from the HTTP API."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import multiprocessing
import signal
import time
import psutil
from app.settings import get_settings
from app.database.store import Store

def child_entry(analysis_id,settings_dict):
    from app.agents.orchestrator import run_child
    run_child(analysis_id,settings_dict)

def stop_process(process):
    try:
        parent=psutil.Process(process.pid)
        for child in parent.children(recursive=True):child.kill()
        parent.kill()
    except psutil.NoSuchProcess:pass
    process.join(timeout=5)

def supervise(analysis_id,settings,store):
    record=store.analysis(analysis_id);budget=min(record['request']['time_budget_seconds'],settings.max_job_seconds)
    process=multiprocessing.get_context('spawn').Process(target=child_entry,args=(analysis_id,settings.model_dump(mode='json')))
    process.start();started=time.monotonic()
    try:
        while process.is_alive():
            record=store.analysis(analysis_id)
            reason=None;status=None
            if record['cancel_requested']:reason='Cancelled by user.';status='cancelled'
            elif time.monotonic()-started>budget:reason=f'Time limit of {budget} seconds exceeded. Completed checkpoints are retained.';status='timed_out'
            else:
                try:
                    parent=psutil.Process(process.pid);memory=parent.memory_info().rss+sum(p.memory_info().rss for p in parent.children(recursive=True))
                    if memory>settings.max_memory_mb*1024**2:reason=f'Memory limit of {settings.max_memory_mb} MB exceeded.';status='failed'
                except psutil.NoSuchProcess:pass
            if reason:
                stop_process(process);store.update(analysis_id,status=status,error=reason);store.event(analysis_id,status,reason);return
            store.update(analysis_id,heartbeat=time.time());process.join(timeout=.5)
        process.join()
        if store.analysis(analysis_id)['status']=='running':store.update(analysis_id,status='failed',error=f'Analysis process exited unexpectedly ({process.exitcode}). Retry to resume checkpoints.')
    except BaseException:
        stop_process(process);store.update(analysis_id,status='interrupted',error='Worker stopped. Retry to resume checkpoints.');raise

def main():
    def shutdown(signum, frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown)
    settings=get_settings();settings.initialize();store=Store(settings.db_path);store.initialize()
    print('AutoDS worker ready.',flush=True)
    while True:
        store.recover();analysis_id=store.claim()
        if analysis_id:supervise(analysis_id,settings,store)
        else:time.sleep(1)

if __name__=='__main__':main()
