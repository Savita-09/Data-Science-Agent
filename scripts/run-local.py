"""Start the API and supervised training worker using the current Python environment."""
import os
import argparse
import subprocess
import sys
import psutil
from pathlib import Path
from time import sleep
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--env-file',type=Path,help='Use only this environment file, for an isolated backend workspace.')
    args=parser.parse_args()
    if not 1<=args.port<=65535:parser.error('--port must be between 1 and 65535')
    if args.env_file:
        if not args.env_file.is_file():parser.error('--env-file must name an existing file')
        load_dotenv(args.env_file,override=True)
        os.environ['PYTHON_DOTENV_DISABLED']='1'
    processes=[]
    try:
        for command in ([sys.executable,'-m','app.worker'],[sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port',str(args.port),'--no-proxy-headers','--no-access-log']):
            processes.append(subprocess.Popen(command,cwd=ROOT/'backend',env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}))
        print(f'AutoDS Studio: http://127.0.0.1:{args.port} (build the frontend first). Ctrl+C stops both services.',flush=True)
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
