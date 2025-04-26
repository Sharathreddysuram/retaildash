from sqlalchemy import create_engine
import pandas as pd
import urllib

params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=sharathfinal.database.windows.net;"
    "DATABASE=finalproject;"
    "UID=siddu;"
    "PWD=Cloud@123"
)

conn_str = f"mssql+pyodbc:///?odbc_connect={params}"
engine = create_engine(conn_str)

# Try pulling one table
hh = pd.read_sql_table("Households", engine)
print(hh.head())
