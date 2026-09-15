import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

def dumps(value):
    return json.dumps(value, allow_nan=False, default=lambda v: v.item() if hasattr(v, 'item') else str(v))

class Store:
    def __init__(self, path): self.path = Path(path)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA busy_timeout=30000')
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback(); raise
        finally: connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
            CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha TEXT NOT NULL, metadata TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id), request TEXT NOT NULL, fingerprint TEXT NOT NULL, status TEXT NOT NULL, stage TEXT, error TEXT, result TEXT, created REAL NOT NULL, updated REAL NOT NULL, heartbeat REAL, cancel_requested INTEGER NOT NULL DEFAULT 0, attempt INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS analyses_queue ON analyses(status, created);
            CREATE TABLE IF NOT EXISTS checkpoints (analysis_id TEXT NOT NULL REFERENCES analyses(id), stage TEXT NOT NULL, result TEXT NOT NULL, duration REAL NOT NULL, completed REAL NOT NULL, PRIMARY KEY(analysis_id,stage));
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT NOT NULL REFERENCES analyses(id), stage TEXT, message TEXT NOT NULL, created REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS events_analysis ON events(analysis_id,id);
            CREATE TABLE IF NOT EXISTS chat (id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT NOT NULL REFERENCES analyses(id), question TEXT NOT NULL, answer TEXT NOT NULL, source TEXT NOT NULL, created REAL NOT NULL);
            ''')

    def dataset(self, dataset_id):
        with self.connect() as db: row = db.execute('SELECT * FROM datasets WHERE id=?', (dataset_id,)).fetchone()
        if not row: return None
        result = dict(row); result['metadata'] = json.loads(result['metadata']); return result

    def add_dataset(self, dataset_id, filename, sha, metadata):
        with self.connect() as db: db.execute('INSERT INTO datasets VALUES (?,?,?,?,?)', (dataset_id, filename, sha, dumps(metadata), time.time()))

    def enqueue(self, analysis_id, dataset_id, request, fingerprint, max_queue):
        now = time.time()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT count(*) FROM analyses WHERE status IN ('queued','running')").fetchone()[0] >= max_queue: raise ValueError('The training queue is full. Try again after a job completes.')
            db.execute('INSERT INTO analyses(id,dataset_id,request,fingerprint,status,created,updated) VALUES (?,?,?,?,?,?,?)', (analysis_id,dataset_id,dumps(request),fingerprint,'queued',now,now))
        self.event(analysis_id, 'queued', 'Analysis added to the durable queue.')

    def analysis(self, analysis_id):
        with self.connect() as db:
            row = db.execute('SELECT a.*, d.filename AS dataset_filename FROM analyses a JOIN datasets d ON d.id=a.dataset_id WHERE a.id=?', (analysis_id,)).fetchone()
            if not row: return None
            result = dict(row)
            result['checkpoints'] = [dict(r) for r in db.execute('SELECT * FROM checkpoints WHERE analysis_id=? ORDER BY completed', (analysis_id,))]
        result['request'] = json.loads(result['request']); result['result'] = json.loads(result['result']) if result['result'] else None
        for cp in result['checkpoints']: cp['result'] = json.loads(cp['result'])
        return result

    def list_analyses(self):
        with self.connect() as db: return [dict(r) for r in db.execute('SELECT a.id,a.dataset_id,d.filename AS dataset_filename,a.status,a.stage,a.error,a.created,a.updated,a.request FROM analyses a JOIN datasets d ON d.id=a.dataset_id ORDER BY a.created DESC LIMIT 100')]

    def event(self, analysis_id, stage, message):
        with self.connect() as db: db.execute('INSERT INTO events(analysis_id,stage,message,created) VALUES (?,?,?,?)', (analysis_id,stage,str(message)[:2000],time.time()))

    def events(self, analysis_id, after=0):
        with self.connect() as db: return [dict(r) for r in db.execute('SELECT * FROM events WHERE analysis_id=? AND id>? ORDER BY id LIMIT 200', (analysis_id,after))]

    def update(self, analysis_id, **values):
        allowed = {'status','stage','error','result','heartbeat','cancel_requested'}
        if set(values) - allowed: raise ValueError('Invalid persistence field.')
        values['updated'] = time.time()
        if 'result' in values and values['result'] is not None: values['result'] = dumps(values['result'])
        with self.connect() as db: db.execute(f"UPDATE analyses SET {','.join(k+'=?' for k in values)} WHERE id=?", (*values.values(),analysis_id))

    def checkpoint(self, analysis_id, stage, result, duration):
        with self.connect() as db: db.execute('INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?)', (analysis_id,stage,dumps(result),duration,time.time()))

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT id FROM analyses WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE analyses SET status='running',heartbeat=?,updated=?,attempt=attempt+1 WHERE id=?", (time.time(),time.time(),row['id']))
                return row['id']

    def recover(self):
        with self.connect() as db:
            db.execute("UPDATE analyses SET status='interrupted',error='Worker heartbeat expired. Retry to resume from checkpoints.',updated=? WHERE status='running' AND heartbeat<?", (time.time(),time.time()-30))

    def retry(self, analysis_id, max_queue=20):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status FROM analyses WHERE id=?', (analysis_id,)).fetchone()
            if not row or row['status'] not in ('failed','interrupted','cancelled','timed_out'): raise ValueError('Only failed, interrupted, cancelled, or timed-out jobs can be retried.')
            if db.execute("SELECT count(*) FROM analyses WHERE status IN ('queued','running')").fetchone()[0] >= max_queue: raise ValueError('The training queue is full. Retry after a job completes.')
            db.execute("UPDATE analyses SET status='queued',error=NULL,cancel_requested=0,updated=? WHERE id=?", (time.time(),analysis_id))

    def save_chat(self, analysis_id, question, answer, source):
        with self.connect() as db: db.execute('INSERT INTO chat(analysis_id,question,answer,source,created) VALUES (?,?,?,?,?)', (analysis_id,question,answer,source,time.time()))

    def chats(self, analysis_id):
        with self.connect() as db: return [dict(r) for r in db.execute('SELECT question,answer,source,created FROM chat WHERE analysis_id=? ORDER BY id DESC LIMIT 30', (analysis_id,))][::-1]
