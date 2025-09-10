import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client, Client
from datetime import datetime

# --------------------------
# Supabase Connection
# --------------------------
url = "https://sywxhehahunevputgxdd.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"

supabase: Client = create_client(url, key)

# --------------------------
# Fetch Data
# --------------------------
@st.cache_data
def load_data():
    products = supabase.table("product_master").select("*").execute().data
    sales = supabase.table("sales_transactions").select("*").execute().data

    df_products = pd.DataFrame(products)
    df_sales = pd.DataFrame(sales)

    # Convert timestamps
    df_sales["created_at"] = pd.to_datetime(df_sales["created_at"], errors="coerce", utc=True)

    return df_products, df_sales

df_products, df_sales = load_data()

# --------------------------
# Dashboard Layout
# --------------------------
st.title("📊 CPG Sales Dashboard")
st.write("Hershey’s product sales analytics (demo dataset)")

# Sidebar Filters
st.sidebar.header("Filters")
category_filter = st.sidebar.multiselect("Select Product Category", options=df_products["product_category"].unique())
date_range = st.sidebar.date_input("Select Date Range", [df_sales["created_at"].min(), df_sales["created_at"].max()])

# Filter data
if category_filter:
    df_sales = df_sales[df_sales["product_id"].isin(df_products[df_products["product_category"].isin(category_filter)]["product_id"])]
if date_range:
    start_date, end_date = date_range
    df_sales = df_sales[(df_sales["created_at"].dt.date >= start_date) & (df_sales["created_at"].dt.date <= end_date)]

# --------------------------
# Aggregations
# --------------------------
df_merged = df_sales.merge(df_products, on="product_id", how="left")
sales_summary = df_merged.groupby("product_category")["units_sold"].sum().reset_index()

# Show Data
st.subheader("Sales by Category")
st.dataframe(sales_summary)

# Plot
fig, ax = plt.subplots()
ax.bar(sales_summary["product_category"], sales_summary["units_sold"])
ax.set_ylabel("Units Sold")
ax.set_xlabel("Category")
ax.set_title("Total Units Sold by Category")
st.pyplot(fig)