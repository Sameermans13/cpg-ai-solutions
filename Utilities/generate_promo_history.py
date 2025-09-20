# generate_promo_history.py

import pandas as pd
from supabase import create_client, Client
import random
from datetime import date, timedelta

# --- CONFIGURE YOUR SUPABASE CONNECTION ---
# Replace with your actual Supabase URL and PUBLIC ANON KEY
SUPABASE_URL = "https://sywxhehahunevputgxdd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def generate_and_insert_promo_data():
    """
    Generates historical promotion data and creates SQL INSERT statements.
    """
    print("Fetching product and store master data...")
    # Fetch product and store IDs for linking
    products_data = supabase.table("product_master").select("product_id, product_category, product_era").execute().data
    stores_data = supabase.table("store_master").select("store_id, store_channel, store_region").execute().data

    df_products = pd.DataFrame(products_data)
    df_stores = pd.DataFrame(stores_data)

    product_ids = df_products['product_id'].tolist()
    store_ids = df_stores['store_id'].tolist()

    print("Generating promotion data...")

    # --- Define Historical Promotions (2024 & 2025) ---
    promotions = [
        # 2024 Promotions
        {"id": 1, "name": "Spring Fling 2024", "tactic": "Price Reduction", "start": date(2024, 4, 1), "end": date(2024, 4, 14), "spend": 4000, "lift": 0.8},
        {"id": 2, "name": "Summer Blast 2024", "tactic": "In-Store Display", "start": date(2024, 6, 1), "end": date(2024, 6, 30), "spend": 8000, "lift": 1.2},
        {"id": 3, "name": "Back to School 2024", "tactic": "Digital Coupon", "start": date(2024, 8, 15), "end": date(2024, 9, 7), "spend": 2500, "lift": 0.6},
        {"id": 4, "name": "Holiday Gifting 2024", "tactic": "Price Reduction", "start": date(2024, 11, 20), "end": date(2024, 12, 31), "spend": 6000, "lift": 1.0},
        {"id": 5, "name": "New Year Kickoff 2025", "tactic": "In-Store Display", "start": date(2025, 1, 1), "end": date(2025, 1, 14), "spend": 5000, "lift": 0.9},
        
        # 2025 Promotions
        {"id": 6, "name": "Spring Fling 2025", "tactic": "Digital Coupon", "start": date(2025, 4, 1), "end": date(2025, 4, 14), "spend": 4500, "lift": 0.85},
        {"id": 7, "name": "Summer Blast 2025", "tactic": "Price Reduction", "start": date(2025, 6, 1), "end": date(2025, 6, 30), "spend": 7000, "lift": 1.1},
        {"id": 8, "name": "Back to School 2025", "tactic": "In-Store Display", "start": date(2025, 8, 15), "end": date(2025, 9, 7), "spend": 3000, "lift": 0.65},
        {"id": 9, "name": "Holiday Gifting 2025", "tactic": "Digital Coupon", "start": date(2025, 11, 20), "end": date(2025, 12, 31), "spend": 6500, "lift": 1.05},
        {"id": 10, "name": "New Era Product Showcase 2025", "tactic": "In-Store Display", "start": date(2025, 9, 15), "end": date(2025, 9, 30), "spend": 9000, "lift": 1.8} # High impact for New Era
    ]

    promo_master_values = []
    promo_products_values = []
    promo_stores_values = []
    
    current_promo_product_id = 1 # For primary key in promotion_products

    for promo in promotions:
        promo_master_values.append(
            f"({promo['id']}, '{promo['name']}', '{promo['tactic']}', '{promo['start']}', '{promo['end']}', {promo['spend']:.2f}, {promo['lift']:.2f})"
        )

        # Link Products to Promotion
        # Each promo will have a subset of products, with varying discounts and planned lifts
        num_products_in_promo = random.randint(3, min(7, len(product_ids)))
        selected_product_ids = random.sample(product_ids, num_products_in_promo)

        for prod_id in selected_product_ids:
            # Get product details to influence discount/lift
            product_detail = df_products[df_products['product_id'] == prod_id].iloc[0]
            
            discount = round(random.uniform(0.10, 0.30), 2) # Base discount
            if product_detail['product_era'] == 'New Era':
                discount = round(random.uniform(0.20, 0.40), 2) # Higher discount for New Era

            # Placeholder for planned_baseline_units - this would come from the weekly_baselines table in real analysis
            # For data generation, we'll use a plausible average.
            # We'll link actual baselines in the Streamlit app.
            planned_baseline = random.randint(1000, 5000) 

            # Projected lift can vary by product and tactic
            product_lift = promo['lift'] * random.uniform(0.8, 1.2) # Base lift adjusted
            if product_detail['product_era'] == 'New Era':
                product_lift *= random.uniform(1.1, 1.5) # Even higher lift for New Era products

            product_spend = round(promo['spend'] / num_products_in_promo * random.uniform(0.8, 1.2), 2) # Allocate spend

            promo_products_values.append(
                f"({current_promo_product_id}, {promo['id']}, {prod_id}, {discount:.2f}, {planned_baseline}, {product_lift:.2f}, {product_spend:.2f})"
            )
            current_promo_product_id += 1
        
        # Link Stores to Promotion
        # Some promos are national, some are channel-specific, some are single-store.
        if random.random() < 0.3: # ~30% chance for a broad/national promo
            if random.random() < 0.5: # Half of broad are Mass Retail, other half Grocery
                relevant_stores = df_stores[df_stores['store_channel'] == 'Mass Retail']['store_id'].tolist()
            else:
                relevant_stores = df_stores[df_stores['store_channel'] == 'Grocery']['store_id'].tolist()
        else: # ~70% chance for a smaller, random subset of stores
            num_stores_in_promo = random.randint(1, min(5, len(store_ids)))
            relevant_stores = random.sample(store_ids, num_stores_in_promo)

        for store_id_in_promo in relevant_stores:
            promo_stores_values.append(
                f"({promo['id']}, {store_id_in_promo})"
            )

    # --- Generate SQL INSERT Statements ---
    print("\n--- COPY AND RUN THE SQL BELOW IN YOUR SUPABASE EDITOR ---")
    
    # 1. Clear existing data
    sql_clear = """
    TRUNCATE public.promotion_products RESTART IDENTITY CASCADE;
    TRUNCATE public.promotion_stores RESTART IDENTITY CASCADE;
    TRUNCATE public.promotions_master RESTART IDENTITY CASCADE;
    """
    print(sql_clear)

    # 2. Insert into promotions_master
    sql_master = "INSERT INTO public.promotions_master (promotion_id, promotion_name, promotion_tactic, start_date, end_date, total_planned_spend, projected_unit_lift) VALUES\n"
    sql_master += ",\n".join(promo_master_values)
    sql_master += ";\n"
    print(sql_master)

    # 3. Insert into promotion_products
    sql_products = "INSERT INTO public.promotion_products (id, promotion_id, product_id, discount_percentage, planned_baseline_units, projected_unit_lift, planned_product_spend) VALUES\n"
    sql_products += ",\n".join(promo_products_values)
    sql_products += ";\n"
    print(sql_products)

    # 4. Insert into promotion_stores
    sql_stores = "INSERT INTO public.promotion_stores (promotion_id, store_id) VALUES\n"
    sql_stores += ",\n".join(promo_stores_values)
    sql_stores += ";\n"
    print(sql_stores)
    
    print("\n--- END OF SQL SCRIPT ---")


if __name__ == "__main__":
    generate_and_insert_promo_data()