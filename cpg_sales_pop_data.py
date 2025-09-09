from supabase import create_client, Client
from datetime import datetime, timedelta, timezone
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
#for p in products:
#    supabase.table("product_master").insert(p).execute()

# -------------------------------
# 2️⃣ Prepopulate Sales Transactions
# -------------------------------

# Fetch products and stores
products = supabase.table("product_master").select("product_id, retail_price").execute().data
stores = supabase.table("store_master").select("store_id, store_region").execute().data

sales_data = []

for _ in range(10000):
    product = random.choice(products)
    store = random.choice(stores)
    
    random_days = random.randint(0, 89)
    sale_date = datetime.now(timezone.utc) - timedelta(days=random_days)
    
    units = random.randint(1, 20)
    price = round(random.uniform(0.8, 1.0) * product["retail_price"], 2)
    
    sales_data.append({
        "product_id": product["product_id"],
        "store_id": store["store_id"],
        "units_sold": units,
        "sale_price": price,
        "created_at": sale_date.isoformat(),
        "region": store.get("store_region")
    })

# Insert in batches of 500
batch_size = 500
for i in range(0, len(sales_data), batch_size):
    supabase.table("sales_transactions").insert(sales_data[i:i+batch_size]).execute()

print("Inserted 10,000 sales transactions successfully ✅")