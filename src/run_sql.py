"""Load the cleaned CSV into SQLite and run sql/analysis.sql."""
import re
import sqlite3
from pathlib import Path

import pandas as pd

DB = "data/churn.db"
df = pd.read_csv("data/processed/customers_clean.csv")
con = sqlite3.connect(DB)
df.to_sql("customers", con, if_exists="replace", index=False)

script = Path("sql/analysis.sql").read_text()
for block in [b.strip() for b in script.split(";") if b.strip()]:
    title = re.search(r"-- name: (.*)", block).group(1)
    query = "\n".join(l for l in block.splitlines() if not l.startswith("--"))
    print(f"\n=== {title} ===")
    print(pd.read_sql_query(query, con).to_string(index=False))
con.close()
