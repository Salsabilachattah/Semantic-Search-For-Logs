import os
import platform
import sys
from pathlib import Path
import psycopg2
from psycopg2 import sql

if platform.system() == "Windows":
    hadoop_home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
    if not hadoop_home:
        candidate = Path(__file__).resolve().parent / "hadoop"
        if (candidate / "bin" / "winutils.exe").exists():
            os.environ["HADOOP_HOME"] = str(candidate)
            os.environ["hadoop.home.dir"] = str(candidate)

    # Hadoop on Windows needs native binaries in %HADOOP_HOME%\bin
    hadoop_home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
    if hadoop_home:
        hadoop_bin = str(Path(hadoop_home) / "bin")
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

        # If hadoop.dll is missing, Hadoop's NativeIO will crash with UnsatisfiedLinkError.
        if not (Path(hadoop_home) / "bin" / "hadoop.dll").exists():
            raise RuntimeError(
                "Missing Hadoop native library: expected %HADOOP_HOME%\\bin\\hadoop.dll.\n"
                "Spark/Hadoop on Windows needs BOTH winutils.exe and hadoop.dll to write to the local filesystem.\n"
                "Fix: put hadoop.dll in .\\hadoop\\bin (or in %HADOOP_HOME%\\bin).\n"
                "A common source for Hadoop 3.4.x Windows x64 binaries is: \n"
                "https://github.com/kontext-tech/winutils/tree/master/hadoop-3.4.0-win10-x64/bin"
            )


os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)



from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, split, lower, regexp_replace, col, expr, concat_ws

# ===========================
# 1. DATA CLEANING WITH SPARK
# ===========================
# spark = SparkSession.builder \
#     .appName("LogPipeline") \
#     .config("spark.executor.memory", "4g") \
#     .getOrCreate()

# #load the full dataset
# df = spark.read.text("access.log/access.log")  

# # Regex pattern for the log format
# pattern = r'^(\S+) - - \[(.*?)\] "(.*?)" (\d{3}) (\d+) "(.*?)" "(.*?)"'

# # Extract fields using regex
# df_parsed = df.select(
#     regexp_extract('value', pattern, 1).alias('ip'),
#     regexp_extract('value', pattern, 2).alias('timestamp'),
#     regexp_extract('value', pattern, 3).alias('request'),
#     regexp_extract('value', pattern, 4).alias('status'),
#     regexp_extract('value', pattern, 5).alias('size'),
#     regexp_extract('value', pattern, 6).alias('referrer'),
#     regexp_extract('value', pattern, 7).alias('user_agent')
# )

# # Extract method + URL (robust to malformed request strings)
# # Spark array indexing throws if the index is out of bounds, so we use SQL get() which returns NULL.
# df_parsed = (
#     df_parsed
#     .withColumn("method", expr("get(split(request, ' '), 0)"))
#     .withColumn("url", expr("get(split(request, ' '), 1)"))
# )

# # Clean URL
# df_clean = df_parsed.withColumn(
#     "clean_url",
#     lower(regexp_replace("url", r'\d+', ''))
# )

# # Build semantic message

# df_final = df_clean.withColumn("message", concat_ws(" ", col("method"), col("clean_url")))

# # impossible to process the full dataset on a local machine, so we sample 5% of the data for testing purposes
# df_sample = df_final.sample(0.05)

# # Save to parquet
# df_sample.select("timestamp", "message").write.mode("overwrite").parquet("logs_clean")

# spark.stop()


# =========================
# 2. EMBEDDINGS
# =========================

# import pandas as pd
# from sentence_transformers import SentenceTransformer

# if not Path("logs_clean").exists():
#     raise RuntimeError(
#         "Missing parquet output folder: ./logs_clean\n"
#         "Fix: run the Spark cleaning step (section 1) to generate logs_clean, then rerun this script."
#     )

# df = pd.read_parquet("logs_clean")
# messages = df["message"].astype(str).tolist()

# model = SentenceTransformer("all-MiniLM-L6-v2")

# embeddings = model.encode(
#     messages,
#     batch_size=64,
#     show_progress_bar=True,
# )


# =========================
# 3. POSTGRESQL + PGVECTOR
# =========================


PG_HOST = "localhost"
PG_PORT = "5432"
PG_DBNAME = "logs_db"
# username = os.environ.get("username") or "postgres"
# # password = os.environ.get("password") or "password"
username = "postgres"
password = "password"
def _connect(dbname: str):
    return psycopg2.connect(
        dbname=dbname,
        user=username,
        password=password,
        host=PG_HOST,
        port=PG_PORT,
    )


def _ensure_database_exists(dbname: str) -> None:
    try:
        conn_admin = _connect("postgres")
    except psycopg2.OperationalError as e:
        raise RuntimeError(
            "PostgreSQL is reachable, but the database is missing and the script couldn't connect to the admin database 'postgres' to create it.\n"
            "Fix options:\n"
            "- If you're using Docker: recreate the container with POSTGRES_DB=logs_db\n"
            "- If you're using local Postgres: create the DB manually with: createdb -h localhost -p 5432 -U postgres logs_db"
        ) from e

    try:
        conn_admin.autocommit = True
        with conn_admin.cursor() as cur_admin:
            try:
                cur_admin.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(dbname)))
            except psycopg2.errors.DuplicateDatabase:
                pass
    finally:
        conn_admin.close()


try:
    conn = _connect(PG_DBNAME)
except psycopg2.OperationalError as e:
    message = str(e)
    if "does not exist" in message and "database" in message:
        _ensure_database_exists(PG_DBNAME)
        conn = _connect(PG_DBNAME)
    elif "Connection refused" in message or "refused" in message:
        raise RuntimeError(
            "PostgreSQL connection failed: nothing is listening on localhost:5432.\n"
            "Fix: start Postgres (or run the pgvector Docker image) and retry."
        ) from e
    else:
        raise
cur = conn.cursor()


def _to_pgvector_literal(values) -> str:
    """Convert a Python sequence of numbers to pgvector's text input format.

    psycopg2 sends Python lists as SQL arrays (e.g. numeric[]), but pgvector
    operators like `<->` expect `vector` on both sides.
    """
    return "[" + ",".join(str(float(x)) for x in values) + "]"

# Create extension + table
# cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# cur.execute("""
# CREATE TABLE IF NOT EXISTS logs (
#     id SERIAL PRIMARY KEY,
#     timestamp TEXT,
#     message TEXT,
#     embedding VECTOR(384)
# );
# """)

conn.commit()

# Insert in batches
# for i in range(0, len(messages), 1000):
#     batch_msgs = messages[i:i+1000]
#     batch_embs = embeddings[i:i+1000]
#     batch_ts = df["timestamp"].iloc[i:i+1000]

#     for msg, emb, ts in zip(batch_msgs, batch_embs, batch_ts):
#         cur.execute(
#             "INSERT INTO logs (timestamp, message, embedding) VALUES (%s, %s, %s::vector)",
#             (ts, msg, _to_pgvector_literal(emb))
#         )

#     conn.commit()


# =========================
# 4. CREATE INDEX
# =========================

cur.execute("""
CREATE INDEX IF NOT EXISTS logs_embedding_idx
ON logs USING ivfflat (embedding vector_cosine_ops);
""")
conn.commit()


# =========================
# 5. SEMANTIC SEARCH TEST
# =========================

query = "GET image product"
query_vector = model.encode([query])[0].tolist()
query_vector_literal = _to_pgvector_literal(query_vector)

cur.execute("""
SELECT message
FROM logs
ORDER BY embedding <-> %s::vector
LIMIT 10;
""", (query_vector_literal,))

results = cur.fetchall()

print("\n=== SEARCH RESULTS ===")
for r in results:
    print(r[0])

conn.close()