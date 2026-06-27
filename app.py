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
TABLE_NAME = "semantic_logs"

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


def parse_unique_messages(params) -> bool:
    return params.get("unique_messages", ["1"])[0] == "1"


def semantic_search(params):
    query = params.get("query", ["critical server error"])[0]
    limit = parse_limit(params)
    unique_messages = parse_unique_messages(params)
    vector = to_pgvector_literal(MODEL.encode([query])[0].tolist())

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SET ivfflat.probes = 10;")
            if unique_messages:
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT
                            id,
                            status,
                            timestamp,
                            method,
                            clean_url,
                            embedding <=> %s::vector AS distance,
                            message,
                            ROW_NUMBER() OVER (
                                PARTITION BY message
                                ORDER BY embedding <=> %s::vector, id
                            ) AS message_rank
                        FROM semantic_logs
                    )
                    SELECT id, status, timestamp, method, clean_url, distance, message
                    FROM ranked
                    WHERE message_rank = 1
                    ORDER BY distance
                    LIMIT %s;
                    """,
                    (vector, vector, limit),
                )
            else:
                #  LIMIT %s;, limit
                cur.execute(
                    """
                    SELECT id, status, timestamp, method, clean_url, embedding <=> %s::vector AS distance, message
                    FROM semantic_logs
                    ORDER BY embedding <=> %s::vector
                    """,
                    (vector, vector),
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
    unique_messages = parse_unique_messages(params)

    with connect() as conn:
        with conn.cursor() as cur:
            if unique_messages:
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT
                            id,
                            status,
                            timestamp,
                            method,
                            clean_url,
                            NULL,
                            message,
                            ROW_NUMBER() OVER (
                                PARTITION BY message
                                ORDER BY id
                            ) AS message_rank
                        FROM semantic_logs
                        WHERE message ILIKE %s OR clean_url ILIKE %s
                    )
                    SELECT id, status, timestamp, method, clean_url, NULL, message
                    FROM ranked
                    WHERE message_rank = 1
                    ORDER BY id
                    LIMIT %s;
                    """,
                    (f"%{keyword}%", f"%{keyword}%", limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, status, timestamp, method, clean_url, NULL, message
                    FROM semantic_logs
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
                FROM semantic_logs
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
                FROM semantic_logs
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
    .inputs { display: grid; grid-template-columns: 2fr 1fr 90px 170px; gap: 10px; align-items: end; }
    label { display: grid; gap: 6px; font-size: 13px; font-weight: 700; color: #334155; }
    input { height: 38px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 0 10px; font-size: 15px; }
    .check { display: flex; align-items: center; gap: 8px; height: 38px; }
    .check input { width: 16px; height: 16px; padding: 0; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }
    button { border: 0; border-radius: 6px; background: #2563eb; color: white; padding: 10px 14px; font-weight: 700; cursor: pointer; }
    button:nth-child(3), button:nth-child(4) { background: #0f766e; }
    button:nth-child(5) { background: #475569; }
    #status { margin: 8px 0 12px; color: #475569; min-height: 22px; }
    #chartCard { display: none; margin: 0 0 16px; padding: 18px; background: white; border: 1px solid #dbe3ea; border-radius: 8px; }
    #chartTitle { margin: 0 0 4px; font-size: 18px; }
    #chartHint { margin: 0 0 14px; color: #64748b; font-size: 13px; }
    #chart { width: 100%; overflow-x: auto; }
    #chart svg { display: block; min-width: 680px; width: 100%; height: auto; }
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
      <label class="check">
        <input id="uniqueMessages" type="checkbox" checked>
        Unique messages
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
    <section id="chartCard" aria-live="polite">
      <h2 id="chartTitle"></h2>
      <p id="chartHint"></p>
      <div id="chart"></div>
    </section>
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
    const SVG_NS = 'http://www.w3.org/2000/svg';
    const COLORS = ['#2563eb', '#dc2626', '#0f766e', '#9333ea', '#ea580c', '#0891b2'];

    function svgElement(name, attributes = {}, text = '') {
      const element = document.createElementNS(SVG_NS, name);
      for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
      if (text !== '') element.textContent = text;
      return element;
    }

    function showChart(title, hint, svg) {
      document.getElementById('chartTitle').textContent = title;
      document.getElementById('chartHint').textContent = hint;
      const chart = document.getElementById('chart');
      chart.replaceChildren(svg);
      document.getElementById('chartCard').style.display = 'block';
    }

    function hideChart() {
      document.getElementById('chartCard').style.display = 'none';
      document.getElementById('chart').replaceChildren();
    }

    function drawRecurrentErrors(rows) {
      if (!rows.length) return hideChart();
      const width = 1000;
      const left = 330;
      const right = 70;
      const rowHeight = 42;
      const height = 50 + rows.length * rowHeight;
      const maxCount = Math.max(...rows.map(row => Number(row.metric) || 0), 1);
      const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'Recurrent errors bar chart' });

      rows.forEach((row, index) => {
        const count = Number(row.metric) || 0;
        const y = 20 + index * rowHeight;
        const barWidth = (width - left - right) * count / maxCount;
        const label = `${row.status} · ${row.url || row.message || 'Unknown error'}`;
        svg.appendChild(svgElement('text', { x: left - 12, y: y + 20, 'text-anchor': 'end', 'font-size': 13, fill: '#334155' }, label.length > 43 ? label.slice(0, 40) + '…' : label));
        const bar = svgElement('rect', { x: left, y, width: Math.max(barWidth, 2), height: 26, rx: 4, fill: '#0f766e' });
        bar.appendChild(svgElement('title', {}, `${label}: ${count}`));
        svg.appendChild(bar);
        svg.appendChild(svgElement('text', { x: left + barWidth + 8, y: y + 19, 'font-size': 13, 'font-weight': 700, fill: '#172033' }, String(count)));
      });
      showChart('Most recurrent HTTP errors', 'Frequency by status, URL, and normalized message.', svg);
    }

    function drawTemporalErrors(rows) {
      if (!rows.length) return hideChart();
      const width = 1000, height = 470;
      const margin = { top: 40, right: 35, bottom: 75, left: 65 };
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const days = [...new Set(rows.map(row => row.timestamp))].sort();
      const statuses = [...new Set(rows.map(row => String(row.status)))].sort();
      const values = new Map(rows.map(row => [`${row.timestamp}|${row.status}`, Number(row.metric) || 0]));
      const maxCount = Math.max(...rows.map(row => Number(row.metric) || 0), 1);
      const x = index => margin.left + (days.length === 1 ? plotWidth / 2 : index * plotWidth / (days.length - 1));
      const y = value => margin.top + plotHeight - value * plotHeight / maxCount;
      const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'HTTP errors over time line chart' });

      for (let tick = 0; tick <= 4; tick++) {
        const value = Math.round(maxCount * tick / 4);
        const tickY = y(value);
        svg.appendChild(svgElement('line', { x1: margin.left, y1: tickY, x2: width - margin.right, y2: tickY, stroke: '#e2e8f0' }));
        svg.appendChild(svgElement('text', { x: margin.left - 10, y: tickY + 4, 'text-anchor': 'end', 'font-size': 12, fill: '#64748b' }, String(value)));
      }
      days.forEach((day, index) => {
        const tickX = x(index);
        svg.appendChild(svgElement('text', { x: tickX, y: height - 42, 'text-anchor': 'end', transform: `rotate(-35 ${tickX} ${height - 42})`, 'font-size': 12, fill: '#64748b' }, day));
      });
      statuses.forEach((status, statusIndex) => {
        const color = COLORS[statusIndex % COLORS.length];
        const points = days.map((day, index) => `${x(index)},${y(values.get(`${day}|${status}`) || 0)}`).join(' ');
        svg.appendChild(svgElement('polyline', { points, fill: 'none', stroke: color, 'stroke-width': 3, 'stroke-linejoin': 'round' }));
        days.forEach((day, index) => {
          const value = values.get(`${day}|${status}`) || 0;
          const circle = svgElement('circle', { cx: x(index), cy: y(value), r: 4, fill: color });
          circle.appendChild(svgElement('title', {}, `${day} · HTTP ${status}: ${value}`));
          svg.appendChild(circle);
        });
        const legendX = margin.left + statusIndex * 125;
        svg.appendChild(svgElement('rect', { x: legendX, y: 8, width: 14, height: 14, rx: 2, fill: color }));
        svg.appendChild(svgElement('text', { x: legendX + 20, y: 20, 'font-size': 13, fill: '#334155' }, `HTTP ${status}`));
      });
      showChart('HTTP errors over time', 'Daily error count, separated by HTTP status.', svg);
    }

    async function run(action) {
      const params = new URLSearchParams({
        action,
        query: document.getElementById('query').value,
        keyword: document.getElementById('keyword').value,
        limit: document.getElementById('limit').value,
        unique_messages: document.getElementById('uniqueMessages').checked ? '1' : '0'
      });
      document.getElementById('status').textContent = 'Running...';
      const response = await fetch('/api?' + params.toString());
      const payload = await response.json();
      if (!response.ok) {
        document.getElementById('status').textContent = payload.error || 'Request failed';
        hideChart();
        return;
      }
      if (action === 'errors') drawRecurrentErrors(payload.rows);
      else if (action === 'temporal') drawTemporalErrors(payload.rows);
      else hideChart();
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
