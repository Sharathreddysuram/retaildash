import pandas as pd
from sqlalchemy import create_engine
import urllib

# 1) Your Azure SQL connection details (using pyodbc driver)
username = "siddu"
password = "Cloud@123"
server   = "sharathfinal.database.windows.net"
database = "finalproject"

# 2) Build the connection string (pyodbc)
conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine   = create_engine(conn_str)

# 3) CSV → table mapping (absolute paths)
base_dir = "/Users/sharathreddy/Desktop/final_project"
tables = [
    ("Households",   f"{base_dir}/400_households.csv"),
    ("Transactions", f"{base_dir}/400_transactions.csv"),
    ("Products",     f"{base_dir}/400_products.csv"),
]

# 4) Load each CSV into Azure SQL
for table, path in tables:
    print(f"\nLoading {path} into table {table}…")
    df = pd.read_csv(path)

    # Clean column names
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
          .str.upper()
    )

    # Bulk insert with chunks
    df.to_sql(
        name=table,
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=5000
    )
    print(f" → {table} loaded ({df.shape[0]} rows).")

print("\n✅ All tables loaded successfully!")
