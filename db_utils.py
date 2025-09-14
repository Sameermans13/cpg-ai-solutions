import psycopg2
import pandas as pd
import streamlit as st

# Cache connection for performance
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host="YOUR_SUPABASE_HOST",
        database="postgres",
        user="YOUR_DB_USER",
        password="YOUR_DB_PASSWORD",
        port="5432"
    )

def run_query(query, params=None):
    conn = get_connection()
    return pd.read_sql(query, conn, params=params)