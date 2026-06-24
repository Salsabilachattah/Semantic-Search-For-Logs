import html
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

MODEL = SentenceTransformer(MODEL_NAME)


def to_pgvector_literal(values) -> str:
    return "[" + ",".join(str(float(x)) for x in values) + "]"


def connect():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD,
    )


def semantic_search(query: str, limit: int):
    query_vector = MODEL.encode([query])[0].tolist()
    vector_literal = to_pgvector_literal(query_vector)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SET ivfflat.probes = 10;")
            cur.execute(
                """
                SELECT id, timestamp, embedding <=> %s::vector AS distance, message
                FROM logs
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vector_literal, vector_literal, limit),
            )
            return cur.fetchall()


def keyword_search(keyword: str, limit: int):
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, timestamp, NULL, message
                FROM logs
                WHERE message ILIKE %s
                ORDER BY id
                LIMIT %s;
                """,
                (f"%{keyword}%", limit),
            )
            return cur.fetchall()


def frequent_messages(limit: int):
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT NULL, NULL, COUNT(*) AS count, message
                FROM logs
                GROUP BY message
                ORDER BY count DESC
                LIMIT %s;
                """,
                (limit,),
            )
            return cur.fetchall()


def temporal_evolution(keyword: str, limit: int):
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT NULL, substring(timestamp from 1 for 11) AS day_bucket, COUNT(*) AS count, %s
                FROM logs
                WHERE message ILIKE %s
                GROUP BY day_bucket
                ORDER BY day_bucket
                LIMIT %s;
                """,
                (f"matches keyword: {keyword}", f"%{keyword}%", limit),
            )
            return cur.fetchall()


def database_stats():
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM logs;")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT message) FROM logs;")
            unique_messages = cur.fetchone()[0]
            cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM logs;")
            first_ts, last_ts = cur.fetchone()

    return [
        (None, None, total, "Total logs"),
        (None, None, unique_messages, "Distinct normalized messages"),
        (None, first_ts, None, "First timestamp"),
        (None, last_ts, None, "Last timestamp"),
    ]


def parse_limit(raw_value: str) -> int:
    try:
        return max(1, min(100, int(raw_value)))
    except ValueError:
        return 10


def rows_to_dicts(rows):
    output = []
    for log_id, timestamp, score, message in rows:
        if isinstance(score, float):
            score = round(score, 4)
        output.append(
            {
                "id": "" if log_id is None else log_id,
                "timestamp": "" if timestamp is None else timestamp,
                "score": "" if score is None else score,
                "message": "" if message is None else message,
            }
        )
    return output


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semantic Log Search</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; }
    body { margin: 0; background: #f4f6f8; color: #17202a; }
    header { background: #16324f; color: white; padding: 18px 28px; }
    h1 { margin: 0; font-size: 24px; font-weight: 700; }
    main { padding: 22px 28px; }
    .bar { display: grid; grid-template-columns: 2fr 1fr 90px; gap: 10px; align-items: end; }
    label { display: grid; gap: 6px; font-size: 13px; font-weight: 700; color: #334155; }
    input { height: 38px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 0 10px; font-size: 15px; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }
    button { border: 0; border-radius: 6px; background: #2563eb; color: white; padding: 10px 14px; font-weight: 700; cursor: pointer; }
    button.secondary { background: #0f766e; }
    button.neutral { background: #475569; }
    #status { min-height: 22px; color: #475569; margin: 8px 0 12px; }
    table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #dbe3ea; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e5eaf0; text-align: left; vertical-align: top; }
    th { background: #e8eef5; font-size: 13px; color: #334155; }
    td { font-size: 14px; }
    .id { width: 80px; }
    .timestamp { width: 210px; }
    .score { width: 110px; }
    @media (max-width: 760px) {
      .bar { grid-template-columns: 1fr; }
      main { padding: 16px; }
      table { font-size: 13px; }
      .timestamp, .score { width: auto; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Semantic Log Search</h1>
    <div>Model: all-MiniLM-L6-v2 | PostgreSQL + pgvector</div>
  </header>
  <main>
    <section class="bar">
      <label>Semantic query
        <input id="query" value="GET image product">
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
      <button class="secondary" onclick="run('keyword')">Keyword Search</button>
      <button class="neutral" onclick="run('frequent')">Frequent Messages</button>
      <button class="neutral" onclick="run('temporal')">Temporal Evolution</button>
      <button class="neutral" onclick="run('stats')">Database Stats</button>
    </section>
    <div id="status">Ready.</div>
    <table>
      <thead>
        <tr><th class="id">ID</th><th class="timestamp">Timestamp</th><th class="score">Score / Count</th><th>Message</th></tr>
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
        for (const key of ['id', 'timestamp', 'score', 'message']) {
          const td = document.createElement('td');
          td.textContent = row[key];
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
        query = params.get("query", [""])[0]
        keyword = params.get("keyword", [""])[0]
        limit = parse_limit(params.get("limit", ["10"])[0])

        try:
            if action == "semantic":
                rows = semantic_search(query, limit)
            elif action == "keyword":
                rows = keyword_search(keyword, limit)
            elif action == "frequent":
                rows = frequent_messages(limit)
            elif action == "temporal":
                rows = temporal_evolution(keyword, limit)
            elif action == "stats":
                rows = database_stats()
            else:
                raise ValueError(f"Unknown action: {html.escape(action)}")

            body = {"status": f"{action} returned {len(rows)} rows", "rows": rows_to_dicts(rows)}
            self.send_response(200)
        except Exception as exc:
            body = {"error": str(exc)}
            self.send_response(500)

        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))


if __name__ == "__main__":
    port = int(os.environ.get("WEB_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Open http://127.0.0.1:{port}")
    server.serve_forever()
