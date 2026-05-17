import sqlite3
import pandas as pd

import os
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "inventory.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def load_table(table_name):
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def list_tables():
    conn = get_connection()
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    conn.close()
    return tables

def run_query(query):
    conn = get_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
