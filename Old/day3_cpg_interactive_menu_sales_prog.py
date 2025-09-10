# cpg_interactive_analytics.py
from supabase import create_client, Client
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
import random
import streamlit as st


# --------------------------
# 1. Connect to Supabase
# --------------------------
url = "https://sywxhehahunevputgxdd.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"

supabase: Client = create_client(url, key)

# --------------------------
# 2. Fetch tables into DataFrames
# --------------------------
def fetch_data():
    product_res = supabase.table("product_master").select("*").execute()
    df_product = pd.DataFrame(product_res.data)
    
    store_res = supabase.table("store_master").select("*").execute()
    df_store = pd.DataFrame(store_res.data)
    
    sales_res = supabase.table("sales_transactions").select("*").execute()
    df_sales = pd.DataFrame(sales_res.data)
    
    # Convert datetime
    df_sales["created_at"] = pd.to_datetime(df_sales["created_at"], utc=True)
    
    return df_product, df_store, df_sales

# --------------------------
# 3. Merge tables for analytics
# --------------------------
def merge_tables(df_product, df_store, df_sales):
    df_merged = df_sales.merge(df_product, on="product_id", how="left") \
                        .merge(df_store, on="store_id", how="left")
    return df_merged

# --------------------------
# 4. Compute KPIs
# --------------------------
def compute_kpis(df):
    df['revenue'] = df['units_sold'] * df['sale_price']
    
    # Example aggregations
    revenue_by_category = df.groupby("product_category")['revenue'].sum().reset_index()
    revenue_by_region = df.groupby("region")['revenue'].sum().reset_index()
    top_products = df.groupby("product_name")['revenue'].sum().sort_values(ascending=False).head(10)
    
    return revenue_by_category, revenue_by_region, top_products

# --------------------------
# 5. Plot KPIs
# --------------------------
def plot_kpis(revenue_by_category, revenue_by_region, top_products):
    plt.figure(figsize=(10,5))
    sns.barplot(data=revenue_by_category, x="product_category", y="revenue")
    plt.title("Revenue by Product Category")
    plt.show()
    
    plt.figure(figsize=(10,5))
    sns.barplot(data=revenue_by_region, x="region", y="revenue")
    plt.title("Revenue by Region")
    plt.show()
    
    plt.figure(figsize=(10,5))
    top_products.plot(kind='bar')
    plt.title("Top 10 Products by Revenue")
    plt.ylabel("Revenue")
    plt.show()

# --------------------------
# 6. Interactive CLI
# --------------------------
def interactive_menu():
    while True:
        print("\nCPG Analytics Menu:")
        print("1. Add Product")
        print("2. Record Sale")
        print("3. View Analytics")
        print("4. Exit")
        choice = input("Enter choice (1-4): ")
        
        if choice == "1":
            name = input("Product Name: ")
            category = input("Category: ")
            sub_category = input("Sub-Category: ")
            price = float(input("Retail Price: "))
            supabase.table("product_master").insert({
                "product_name": name,
                "product_category": category,
                "product_sub_category": sub_category,
                "retail_price": price
            }).execute()
            print("Product added successfully!")
        
        elif choice == "2":
            # You can extend with validations
            product_id = int(input("Product ID: "))
            store_id = int(input("Store ID: "))
            units = int(input("Units Sold: "))
            price = float(input("Sale Price: "))
            supabase.table("sales_transactions").insert({
                "product_id": product_id,
                "store_id": store_id,
                "units_sold": units,
                "sale_price": price,
                "created_at": datetime.now(timezone.utc)
            }).execute()
            print("Sale recorded successfully!")
        
        elif choice == "3":
            df_product, df_store, df_sales = fetch_data()
            df_merged = merge_tables(df_product, df_store, df_sales)
            rev_cat, rev_reg, top_prod = compute_kpis(df_merged)
            print("\nRevenue by Category:\n", rev_cat)
            print("\nRevenue by Region:\n", rev_reg)
            print("\nTop 10 Products:\n", top_prod)
            plot_kpis(rev_cat, rev_reg, top_prod)
        
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    interactive_menu()