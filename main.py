import os
import platform
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
from sentence_transformers import SentenceTransformer

from config import load_env


load_env()

MODEL_NAME = os.environ.get("MODEL_NAME_OR_PATH", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
TABLE_NAME = "semantic_logs"
RAW_LOG_PATH = os.environ.get("RAW_LOG_PATH", "access.log/access.log")
CLEAN_LOG_PATH = os.environ.get("CLEAN_LOG_PATH", "logs_clean")
SAMPLE_FRACTION = float(os.environ.get("SAMPLE_FRACTION", "0.05"))
BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "128"))

PG_HOST = os.environ["PG_HOST"]
PG_PORT = os.environ["PG_PORT"]
PG_DBNAME = os.environ["PG_DBNAME"]
PG_USER = os.environ["PG_USER"]
PG_PASSWORD = os.environ["PG_PASSWORD"]


def configure_windows_hadoop() -> None:
    if platform.system() != "Windows":
        return

    hadoop_home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
    if not hadoop_home:
        candidate = Path(__file__).resolve().parent / "hadoop"
        if (candidate / "bin" / "winutils.exe").exists():
            os.environ["HADOOP_HOME"] = str(candidate)
            os.environ["hadoop.home.dir"] = str(candidate)

    hadoop_home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
    if hadoop_home:
        hadoop_bin = str(Path(hadoop_home) / "bin")
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

        if not (Path(hadoop_home) / "bin" / "hadoop.dll").exists():
            raise RuntimeError(
                "Missing Hadoop native library: expected %HADOOP_HOME%\\bin\\hadoop.dll.\n"
                "Spark/Hadoop on Windows needs both winutils.exe and hadoop.dll."
            )


def clean_logs_with_spark() -> None:
    configure_windows_hadoop()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, concat_ws, expr, lower, regexp_extract, regexp_replace, when

    spark = (
        SparkSession.builder.appName("LogSemanticPipeline")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )

    pattern = r'^(\S+) - - \[(.*?)\] "(.*?)" (\d{3}) (\S+) "(.*?)" "(.*?)"'
    raw_df = spark.read.text(RAW_LOG_PATH)

    parsed_df = raw_df.select(
        regexp_extract("value", pattern, 1).alias("ip"),
        regexp_extract("value", pattern, 2).alias("timestamp"),
        regexp_extract("value", pattern, 3).alias("request"),
        regexp_extract("value", pattern, 4).cast("int").alias("status"),
        regexp_extract("value", pattern, 5).alias("size_raw"),
        regexp_extract("value", pattern, 6).alias("referrer"),
        regexp_extract("value", pattern, 7).alias("user_agent"),
    )

    cleaned_df = (
        parsed_df.withColumn("method", expr("get(split(request, ' '), 0)"))
        .withColumn("url", expr("get(split(request, ' '), 1)"))
        .withColumn("size", when(col("size_raw") == "-", None).otherwise(col("size_raw").cast("long")))
        .withColumn("clean_url", lower(regexp_replace(col("url"), r"\d+", "")))
        .withColumn("message", concat_ws(" ", col("method"), col("clean_url")))
        .drop("size_raw")
        .filter(col("timestamp") != "")
        .filter(col("message") != "")
    )

    sampled_df = cleaned_df.sample(SAMPLE_FRACTION, seed=42)
    sampled_df.select(
        "ip",
        "timestamp",
        "method",
        "url",
        "clean_url",
        "status",
        "size",
        "referrer",
        "user_agent",
        "message",
    ).write.mode("overwrite").parquet(CLEAN_LOG_PATH)

    spark.stop()


def connect(dbname: str):
    return psycopg2.connect(
        dbname=dbname,
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
    )


def ensure_database_exists(dbname: str) -> None:
    try:
        conn_admin = connect("postgres")
    except psycopg2.OperationalError as exc:
        raise RuntimeError(
            "Could not connect to the admin database 'postgres'. "
            "Create logs_db manually or check PG_USER/PG_PASSWORD in .env."
        ) from exc

    try:
        conn_admin.autocommit = True
        with conn_admin.cursor() as cur:
            try:
                cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(dbname)))
            except psycopg2.errors.DuplicateDatabase:
                pass
    finally:
        conn_admin.close()


def connect_project_db():
    try:
        return connect(PG_DBNAME)
    except psycopg2.OperationalError as exc:
        message = str(exc)
        if "does not exist" in message and "database" in message:
            ensure_database_exists(PG_DBNAME)
            return connect(PG_DBNAME)
        raise


def to_pgvector_literal(values) -> str:
    return "[" + ",".join(str(float(x)) for x in values) + "]"


def ensure_table(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                ip TEXT,
                timestamp TEXT,
                method TEXT,
                url TEXT,
                clean_url TEXT,
                status INTEGER,
                size BIGINT,
                referrer TEXT,
                user_agent TEXT,
                message TEXT,
                embedding VECTOR(384)
            );
            """
        )
        cur.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding';",
            (TABLE_NAME,),
        )
        embedding_typmod = cur.fetchone()[0]
        if embedding_typmod != EMBEDDING_DIM:
            raise RuntimeError(
                f"{TABLE_NAME}.embedding has dimension {embedding_typmod}, but {MODEL_NAME} needs {EMBEDDING_DIM}.\n"
                f"Run this in psql, then rerun main.py:\nDROP TABLE IF EXISTS {TABLE_NAME};"
            )
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        row_count = cur.fetchone()[0]
    conn.commit()
    return row_count


def insert_rows(conn, df: pd.DataFrame) -> None:
    model = SentenceTransformer(MODEL_NAME)
    messages = df["message"].astype(str).tolist()
    embeddings = model.encode(messages, batch_size=BATCH_SIZE, show_progress_bar=True)

    insert_sql = f"""
        INSERT INTO {TABLE_NAME} (
            ip, timestamp, method, url, clean_url, status, size, referrer, user_agent, message, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector);
    """

    with conn.cursor() as cur:
        for start in range(0, len(df), 1000):
            batch_df = df.iloc[start : start + 1000]
            batch_embeddings = embeddings[start : start + 1000]
            for (_, row), embedding in zip(batch_df.iterrows(), batch_embeddings):
                cur.execute(
                    insert_sql,
                    (
                        row.get("ip"),
                        row.get("timestamp"),
                        row.get("method"),
                        row.get("url"),
                        row.get("clean_url"),
                        None if pd.isna(row.get("status")) else int(row.get("status")),
                        None if pd.isna(row.get("size")) else int(row.get("size")),
                        row.get("referrer"),
                        row.get("user_agent"),
                        row.get("message"),
                        to_pgvector_literal(embedding),
                    ),
                )
            conn.commit()
            print(f"Inserted {min(start + 1000, len(df))}/{len(df)} rows")


def create_indexes(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS semantic_logs_embedding_cosine_idx
            ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops);
            """
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS semantic_logs_status_idx ON {TABLE_NAME} (status);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS semantic_logs_timestamp_idx ON {TABLE_NAME} (timestamp);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS semantic_logs_message_idx ON {TABLE_NAME} (message);")
    conn.commit()


def main() -> int:
    if os.environ.get("RUN_SPARK_CLEANING", "1") == "1":
        clean_logs_with_spark()

    clean_path = Path(CLEAN_LOG_PATH)
    if not clean_path.exists():
        raise RuntimeError("Missing logs_clean. Run Spark cleaning first or set RUN_SPARK_CLEANING=1.")

    df = pd.read_parquet(clean_path)
    print(f"Loaded {len(df)} cleaned logs from {CLEAN_LOG_PATH}")

    conn = connect_project_db()
    try:
        existing_rows = ensure_table(conn)
        if existing_rows:
            print(f"{TABLE_NAME} already contains {existing_rows} rows; skipping embedding and insertion.")
        else:
            insert_rows(conn, df)
        create_indexes(conn)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
