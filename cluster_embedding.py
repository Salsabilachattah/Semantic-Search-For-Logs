import os
import psycopg2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from psycopg2.extras import execute_batch

from config import load_env

# Load variables from .env
load_env()

PG_HOST = os.environ["PG_HOST"]
PG_PORT = os.environ["PG_PORT"]
PG_DBNAME = os.environ["PG_DBNAME"]
PG_USER = os.environ["PG_USER"]
PG_PASSWORD = os.environ["PG_PASSWORD"]

conn = psycopg2.connect(
    host=PG_HOST,
    port=PG_PORT,
    dbname=PG_DBNAME,
    user=PG_USER,
    password=PG_PASSWORD,
)

cur = conn.cursor()

cur.execute("""
SELECT id, embedding
FROM logs
ORDER BY id;
""")

rows = cur.fetchall()


ids = []
embeddings = []

for log_id, vector in rows:
    ids.append(log_id)

    if isinstance(vector, str):
        vector = np.fromstring(
            vector.strip("[]"),
            sep=",",
            dtype=np.float32
        )

    embeddings.append(vector)

embeddings = np.stack(embeddings)

print("Loaded", len(ids), "embeddings")

# Choose number of clusters
k = 50

kmeans = MiniBatchKMeans(
    n_clusters=k,
    random_state=42,
    batch_size=4096,
    n_init="auto"
)

labels = kmeans.fit_predict(embeddings)

updates = list(zip(labels.tolist(), ids))

execute_batch(
    cur,
    """
    UPDATE logs
    SET cluster_id=%s
    WHERE id=%s;
    """,
    updates,
    page_size=5000
)

conn.commit()

print("Clusters saved.")

cur.close()
conn.close()