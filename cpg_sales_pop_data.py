from supabase import create_client, Client
from datetime import datetime, timedelta
import random


url = "https://sywxhehahunevputgxdd.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"

supabase = create_client(url, key)

# -------------------------------
# 1️⃣ Prepopulate Product Master
# -------------------------------
products = [
    {"product_name": "Hershey's Milk Chocolate Bar", "product_category": "Chocolate", "product_sub_category": "Chocolate Bar", "retail_price": 1.5},
    {"product_name": "Hershey's Dark Chocolate Bar", "product_category": "Chocolate", "product_sub_category": "Chocolate Bar", "retail_price": 1.75},
    {"product_name": "Reese's Peanut Butter Cups", "product_category": "Chocolate", "product_sub_category": "Cups", "retail_price": 1.25},
    {"product_name": "Hershey's Kisses", "product_category": "Chocolate", "product_sub_category": "Bite Size", "retail_price": 3.0},
    {"product_name": "KitKat 4-Pack", "product_category": "Chocolate", "product_sub_category": "Wafer Bar", "retail_price": 2.5},
    {"product_name": "Twix 2-Pack", "product_category": "Chocolate", "product_sub_category": "Wafer Bar", "retail_price": 2.0},
    {"product_name": "Hershey's Syrup", "product_category": "Baking", "product_sub_category": "Syrup", "retail_price": 3.5},
    {"product_name": "Hershey's Cocoa Powder", "product_category": "Baking", "product_sub_category": "Powder", "retail_price": 4.0},
]

# Insert products
for p in products:
    supabase.table("product_master").insert(p).execute()

# -------------------------------
# 2️⃣ Prepopulate Sales Transactions
# -------------------------------
product_records = supabase.table("product_master").select("product_id, product_name, retail_price").execute().data
regions = ["North", "South", "East", "West"]
stores = [1,2,3,4]

start_date = datetime(2025, 9, 1)

transactions = []
for i in range(500):  # generate 500 transactions
    prod = random.choice(product_records)
    product_id = prod["product_id"]
    retail_price = prod["retail_price"]
    units_sold = random.randint(50, 200)
    store_id = random.choice(stores)
    region = random.choice(regions)
    created_at = start_date + timedelta(days=random.randint(0,10), hours=random.randint(8,18))
    transactions.append({
        "product_id": product_id,
        "units_sold": units_sold,
        "sale_price": retail_price,
        "store_id": store_id,
        "region": region,
        "created_at": created_at.isoformat()
    })

# Insert transactions in batches of 50
for i in range(0, len(transactions), 50):
    supabase.table("sales_transactions").insert(transactions[i:i+50]).execute()

print("Prepopulation complete: products + sales transactions")