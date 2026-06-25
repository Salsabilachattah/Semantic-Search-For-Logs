import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import psycopg2
from sentence_transformers import SentenceTransformer

from config import load_env


load_env()

MODEL_NAME = os.environ.get("MODEL_NAME_OR_PATH", "all-MiniLM-L6-v2")
PG_HOST = os.environ["PG_HOST"]
PG_PORT = os.environ["PG_PORT"]
PG_DBNAME = os.environ["PG_DBNAME"]
PG_USER = os.environ["PG_USER"]
PG_PASSWORD = os.environ["PG_PASSWORD"]
TABLE_NAME = "sematic_logs"

MODEL = SentenceTransformer(MODEL_NAME)


def connect():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD,
    )


def to_pgvector_literal(values) -> str:
    return "[" + ",".join(str(float(x)) for x in values) + "]"


def parse_limit(params) -> int:
    try:
        return max(1, min(100, int(params.get("limit", ["10"])[0])))
    except ValueError:
        return 10


def semantic_search(params):
    query = params.get("query", ["critical server error"])[0]
    limit = parse_limit(params)
    vector = to_pgvector_literal(MODEL.encode([query])[0].tolist())

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SET ivfflat.probes = 10;")
            cur.execute(
                """
                SELECT id, status, timestamp, method, clean_url, embedding <=> %s::vector AS distance, message
                FROM sematic_logs
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vector, vector, limit),
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "status": row[1],
            "timestamp": row[2],
            "method": row[3],
            "url": row[4],
            "metric": round(row[5], 4),
            "message": row[6],
        }
        for row in rows
    ]


def keyword_search(params):
    keyword = params.get("keyword", ["error"])[0]
    limit = parse_limit(params)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, timestamp, method, clean_url, NULL, message
                FROM sematic_logs
                WHERE message ILIKE %s OR clean_url ILIKE %s
                ORDER BY id
                LIMIT %s;
                """,
                (f"%{keyword}%", f"%{keyword}%", limit),
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "status": row[1],
            "timestamp": row[2],
            "method": row[3],
            "url": row[4],
            "metric": "",
            "message": row[6],
        }
        for row in rows
    ]


def recurrent_errors(params):
    limit = parse_limit(params)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT NULL, status, NULL, NULL, clean_url, COUNT(*) AS frequency, message
                FROM sematic_logs
                WHERE status >= 400
                GROUP BY status, clean_url, message
                ORDER BY frequency DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        {
            "id": "",
            "status": row[1],
            "timestamp": "",
            "method": "",
            "url": row[4],
            "metric": row[5],
            "message": row[6],
        }
        for row in rows
    ]


def temporal_errors(params):
    limit = parse_limit(params)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT NULL, status, substring(timestamp from 1 for 11) AS day_bucket, NULL, NULL, COUNT(*), 'HTTP errors'
                FROM sematic_logs
                WHERE status >= 400
                GROUP BY status, day_bucket
                ORDER BY day_bucket, status
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        {
            "id": "",
            "status": row[1],
            "timestamp": row[2],
            "method": "",
            "url": "",
            "metric": row[5],
            "message": row[6],
        }
        for row in rows
    ]


def database_stats(_params):
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
            total = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(DISTINCT message) FROM {TABLE_NAME};")
            distinct_messages = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE status >= 400;")
            error_count = cur.fetchone()[0]
            cur.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {TABLE_NAME};")
            first_ts, last_ts = cur.fetchone()

    return [
        {"id": "", "status": "", "timestamp": "", "method": "", "url": "", "metric": total, "message": "Total logs"},
        {
            "id": "",
            "status": "",
            "timestamp": "",
            "method": "",
            "url": "",
            "metric": distinct_messages,
            "message": "Distinct normalized messages",
        },
        {"id": "", "status": "", "timestamp": "", "method": "", "url": "", "metric": error_count, "message": "HTTP errors"},
        {"id": "", "status": "", "timestamp": first_ts or "", "method": "", "url": "", "metric": "", "message": "First log"},
        {"id": "", "status": "", "timestamp": last_ts or "", "method": "", "url": "", "metric": "", "message": "Last log"},
    ]


ACTIONS = {
    "semantic": semantic_search,
    "keyword": keyword_search,
    "errors": recurrent_errors,
    "temporal": temporal_errors,
    "stats": database_stats,
}


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semantic Log Search</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f5f7fa; color: #172033; }
    header { padding: 18px 28px; background: #15324f; color: white; }
    h1 { margin: 0 0 4px; font-size: 24px; }
    main { padding: 22px 28px; }
    .inputs { display: grid; grid-template-columns: 2fr 1fr 90px; gap: 10px; }
    label { display: grid; gap: 6px; font-size: 13px; font-weight: 700; color: #334155; }
    input { height: 38px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 0 10px; font-size: 15px; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }
    button { border: 0; border-radius: 6px; background: #2563eb; color: white; padding: 10px 14px; font-weight: 700; cursor: pointer; }
    button:nth-child(3), button:nth-child(4) { background: #0f766e; }
    button:nth-child(5) { background: #475569; }
    #status { margin: 8px 0 12px; color: #475569; min-height: 22px; }
    table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #dbe3ea; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e5eaf0; text-align: left; vertical-align: top; }
    th { background: #e8eef5; color: #334155; font-size: 13px; }
    td { font-size: 14px; }
    .small { width: 88px; }
    .time { width: 190px; }
    .url { width: 260px; }
    @media (max-width: 800px) { .inputs { grid-template-columns: 1fr; } main { padding: 16px; } }
  </style>
</head>
<body>
  <header>
    <h1>Semantic Log Search</h1>
    <div></div>
  </header>
  <main>
    <section class="inputs">
      <label>Semantic query
        <input id="query" value="critical server error">
      </label>
      <label>Keyword
        <input id="keyword" value="error">
      </label>
      <label>Limit
        <input id="limit" type="number" min="1" max="100" value="10">
      </label>
    </section>
    <section class="actions">
      <button onclick="run('semantic')">Semantic Search</button>
      <button onclick="run('keyword')">Keyword Search</button>
      <button onclick="run('errors')">Recurrent Errors</button>
      <button onclick="run('temporal')">Errors Over Time</button>
      <button onclick="run('stats')">Database Stats</button>
    </section>
    <div id="status">Ready.</div>
    <table>
      <thead>
        <tr>
          <th class="small">ID</th><th class="small">Status</th><th class="time">Timestamp</th>
          <th class="small">Method</th><th class="url">URL</th><th class="small">Score/Count</th><th>Message</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </main>
  <script>
    async function run(action) {
      const params = new URLSearchParams({
        action,
        query: document.getElementById('query').value,
        keyword: document.getElementById('keyword').value,
        limit: document.getElementById('limit').value
      });
      document.getElementById('status').textContent = 'Running...';
      const response = await fetch('/api?' + params.toString());
      const payload = await response.json();
      if (!response.ok) {
        document.getElementById('status').textContent = payload.error || 'Request failed';
        return;
      }
      const body = document.getElementById('rows');
      body.innerHTML = '';
      for (const row of payload.rows) {
        const tr = document.createElement('tr');
        for (const key of ['id', 'status', 'timestamp', 'method', 'url', 'metric', 'message']) {
          const td = document.createElement('td');
          td.textContent = row[key] ?? '';
          tr.appendChild(td);
        }
        body.appendChild(tr);
      }
      document.getElementById('status').textContent = payload.status;
    }
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode("utf-8"))
            return

        if parsed.path == "/api":
            self.handle_api(parsed.query)
            return

        self.send_error(404)

    def handle_api(self, query_string):
        params = parse_qs(query_string)
        action = params.get("action", ["semantic"])[0]

        try:
            if action not in ACTIONS:
                raise ValueError(f"Unknown action: {action}")
            rows = ACTIONS[action](params)
            payload = {"status": f"{action} returned {len(rows)} rows", "rows": rows}
            self.send_response(200)
        except Exception as exc:
            payload = {"error": str(exc)}
            self.send_response(500)

        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))


if __name__ == "__main__":
    port = int(os.environ.get("WEB_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Open http://127.0.0.1:{port}")
    server.serve_forever()
