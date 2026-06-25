import argparse
import os

import psycopg2
from sentence_transformers import SentenceTransformer
from config import load_env

load_env()

TABLE_NAME = "sematic_logs"


def _to_pgvector_literal(values) -> str:
    """Convert a Python sequence of numbers to pgvector's text input format."""
    return "[" + ",".join(str(float(x)) for x in values) + "]"


def _connect(*, host: str, port: str, dbname: str, user: str, password: str):
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic search over logs stored in Postgres+pgvector")
    parser.add_argument("--query", default="22/Jan/2019")
    parser.add_argument("--limit", type=int, default=10, help="Number of results")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME_OR_PATH", "all-MiniLM-L6-v2"),
        help="SentenceTransformer model name or local path (or set MODEL_NAME_OR_PATH)",
    )

    parser.add_argument("--host", default=os.environ["PG_HOST"])
    parser.add_argument("--port", default=os.environ["PG_PORT"])
    parser.add_argument("--dbname", default=os.environ["PG_DBNAME"])
    parser.add_argument("--user", default=os.environ["PG_USER"])
    parser.add_argument("--password", default=os.environ["PG_PASSWORD"])

    parser.add_argument(
        "--distance",
        choices=["cosine", "l2"],
        default=os.environ.get("PGVECTOR_DISTANCE", "cosine"),
        help="Distance metric used for ranking (default: cosine)",
    )

    parser.add_argument(
        "--ensure-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ensure an ivfflat index exists before querying (default: enabled)",
    )

    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    query_vector = model.encode([args.query])[0].tolist()
    query_vector_literal = _to_pgvector_literal(query_vector)

    op = "<=>" if args.distance == "cosine" else "<->"
    opclass = "vector_cosine_ops" if args.distance == "cosine" else "vector_l2_ops"
    index_name = "sematic_logs_embedding_cosine_idx" if args.distance == "cosine" else "sematic_logs_embedding_l2_idx"

    with _connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    ) as conn:
        with conn.cursor() as cur:
            # Optional: tune ivfflat probing if you created an ivfflat index.
            # Higher probes = better recall, slower queries.
            try:
                cur.execute("SET ivfflat.probes = 10;")
            except psycopg2.Error:
                # If ivfflat isn't installed/used, ignore.
                conn.rollback()

            if args.ensure_index:
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {TABLE_NAME} USING ivfflat (embedding {opclass});
                    """
                )

            cur.execute(
                f"""
                SELECT id, status, timestamp, method, clean_url, message
                FROM {TABLE_NAME}
                ORDER BY embedding {op} %s::vector
                LIMIT %s;
                """,
                (query_vector_literal, args.limit),
            )
            rows = cur.fetchall()

    print("\n=== SEARCH RESULTS ===")
    for log_id, status, timestamp, method, clean_url, message in rows:
        print(f"[{log_id}] {status} {timestamp} {method} {clean_url} | {message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
