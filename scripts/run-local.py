"""Start the API and supervised training worker using the current Python environment."""
import os
import subprocess
import sys
import psutil
from pathlib import Path
from time import sleep

ROOT=Path(__file__).resolve().parents[1]
def main():
    processes=[]
    try:
        for command in ([sys.executable,'-m','app.worker'],[sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000','--no-proxy-headers','--no-access-log']):
            processes.append(subprocess.Popen(command,cwd=ROOT/'backend',env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}))
        print('AutoDS Studio: http://127.0.0.1:8000 (build the frontend first). Ctrl+C stops both services.',flush=True)
        while all(process.poll() is None for process in processes):sleep(.5)
        if any(process.poll() not in (None,0) for process in processes):raise RuntimeError('A service exited unexpectedly. Inspect the output above.')
    except KeyboardInterrupt:pass
    finally:
        descendants=[]
        for process in processes:
            if process.poll() is None:
                try:descendants.extend(psutil.Process(process.pid).children(recursive=True))
                except psutil.NoSuchProcess:pass
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for child in descendants:
            try:child.kill()
            except psutil.NoSuchProcess:pass

if __name__=='__main__':main()
