import pandas as pd
import numpy as np
from datetime import date, timedelta
from supabase import create_client, Client
import random

# --- CONFIGURATION ---
SUPABASE_URL = "https://sywxhehahunevputgxdd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"
START_DATE = date(2023, 9, 1) 
END_DATE = date(2025, 9, 14) 
BATCH_SIZE = 15000 # Number of transactions to insert per batch

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def generate_plan_driven_data():
    print("Starting comprehensive, plan-driven data generation...")
    
    # --- DYNAMICALLY FETCH ALL MASTER DATA ---
    print("Fetching master data from Supabase...")
    products_df = pd.DataFrame(supabase.table('product_master').select('*').execute().data)
    stores_df = pd.DataFrame(supabase.table('store_master').select('store_id, store_region').execute().data)
    
    # --- FETCH THE COMPLETE PROMOTION PLAN ---
    promotions_df = pd.DataFrame(supabase.table('promotions_master').select('*').execute().data)
    promo_products_df = pd.DataFrame(supabase.table('promotion_products').select('*').execute().data)
    promo_stores_df = pd.DataFrame(supabase.table('promotion_stores').select('*').execute().data)
    
    # Create a single, detailed promotion plan table for easy lookup
    # Ensure correct 'on' columns for merging
    promo_plan = promotions_df.merge(promo_products_df, on='promotion_id')
    promo_plan = promo_plan.merge(promo_stores_df, on='promotion_id')
    promo_plan['start_date'] = pd.to_datetime(promo_plan['start_date'])
    promo_plan['end_date'] = pd.to_datetime(promo_plan['end_date'])
    
    print(f"Promotion plan loaded with {len(promo_plan)} product/store combinations.")

    # --- GENERATION LOGIC ---
    all_transactions = []
    date_range = pd.to_datetime(pd.date_range(START_DATE, END_DATE))

    for _, product in products_df.iterrows():
        print(f"Generating data for: {product['product_name']}")
        for _, store in stores_df.iterrows():
            store_id = store['store_id']
            
            for day in date_range:
                # --- 1. CALCULATE REALISTIC BASELINE ---
                base_demand = product['retail_price'] * 4 # Simple baseline
                
                # Day of Week Pattern
                dow_multiplier = 1.8 if day.dayofweek >= 4 else 1.0 # Weekend boost
                
                # Product Category Seasonality
                month = day.month
                season_multiplier = 1.0
                if product['product_category'] == 'Ice Cream':
                    if 6 <= month <= 8: season_multiplier = 2.5
                    elif month in [12, 1, 2]: season_multiplier = 0.5
                else: # Candy & Chocolate
                    if month in [2, 10, 12]: season_multiplier = 1.8
                
                # Product Era Trend
                era_multiplier = 1.0
                if product['product_era'] == 'New Era':
                    days_since_start = (day - date_range[0]).days
                    era_multiplier = 1 + (days_since_start / 365) * 0.1 # 10% annual growth
                
                daily_units_baseline = base_demand * dow_multiplier * season_multiplier * era_multiplier
                daily_units_final = max(1, int(daily_units_baseline * random.uniform(0.9, 1.1)))
                
                on_promotion_flag = False
                sale_price = product['retail_price']

                # --- 2. LOOKUP & APPLY PROMOTION PLAN ---
                # IMPORTANT: Use 'product_id' and 'store_id' as they appear in the merged promo_plan
                active_promo = promo_plan[
                    (promo_plan['product_id'] == product['product_id']) & # Using 'product_id' from promo_products_df
                    (promo_plan['store_id'] == store_id) &
                    (promo_plan['start_date'] <= day) &
                    (promo_plan['end_date'] >= day)
                ]
                
                if not active_promo.empty:
                    promo_details = active_promo.iloc[0]
                    on_promotion_flag = True
                    
                    # --- ADD THIS DEBUG LINE ---
                    print(f"DEBUG: Columns in promo_details for product {product['product_name']} (ID: {product['product_id']}) and store {store_id}:")
                    print(promo_details.index.tolist())
                    # --- END DEBUG LINE ---
                    
                    # Use the PLANNED lift - the raw column name from promotion_products is likely 'projected_unit_lift'
                    lift_multiplier = 1 + promo_details['projected_unit_lift_y']

                    lift_variance = random.uniform(0.9, 1.1) # Add some real-world variance
                    daily_units_final = int(daily_units_baseline * lift_multiplier * lift_variance)
                    sale_price = product['retail_price'] * (1 - promo_details['discount_percentage'])

                # --- 3. SIMULATE TRANSACTIONS ---
                num_transactions = random.randint(3, 12)
                if daily_units_final > 0:
                    # Distribute daily_units_final into num_transactions, ensuring sum is daily_units_final
                    transaction_units = np.random.multinomial(int(daily_units_final), np.ones(num_transactions)/num_transactions).tolist()
                    
                    # Account for potential floating point discrepancies and ensure all units are assigned
                    current_sum = sum(transaction_units)
                    diff = int(daily_units_final) - current_sum
                    if diff > 0:
                        for _ in range(diff):
                            transaction_units[random.randint(0, num_transactions - 1)] += 1
                    elif diff < 0:
                        for _ in range(abs(diff)):
                            idx = random.randint(0, num_transactions - 1)
                            if transaction_units[idx] > 0:
                                transaction_units[idx] -= 1
                    
                    for units in transaction_units:
                        if units > 0:
                            all_transactions.append({
                                "created_at": day, "product_id": product['product_id'], "store_id": store_id, 
                                "units_sold": int(units), "on_promotion": on_promotion_flag, 
                                "sale_price": round(sale_price, 2)
                            })
                            
    final_df = pd.DataFrame(all_transactions)
    print(f"Generated {len(final_df)} total transactions.")
    return final_df

def upload_to_supabase(df, table_name):
    print(f"\nUploading {len(df)} records to {table_name}...")
    try:
        # Assuming 'transaction_id' is an auto-incrementing PK
        # We delete all existing records to ensure a fresh dataset
        # This will reset auto-increment if table is truncated, which is good for clean slate
        supabase.table(table_name).delete().neq('transaction_id', 0).execute() # Safer way to delete all
        print(f"Cleared existing data from {table_name}.")
    except Exception as e:
        print(f"Could not clear table {table_name}: {e}")
        print("Continuing without clearing. This may result in duplicate data if not handled by Supabase.")

    if 'created_at' in df.columns:
        df['created_at'] = df['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE].to_dict(orient='records')
        try:
            supabase.table(table_name).insert(batch).execute()
            print(f"Uploaded {min(i + BATCH_SIZE, len(df))} / {len(df)} records.")
        except Exception as e:
            print(f"Error uploading batch: {e}")
            break

if __name__ == "__main__":
    confirm = input("This will DELETE ALL data in 'sales_transactions' and REGENERATE it. Are you sure? (yes/no): ")
    if confirm.lower() == 'yes':
        new_sales_data = generate_plan_driven_data()
        if not new_sales_data.empty:
            upload_to_supabase(new_sales_data, 'sales_transactions')
            print("\nData generation complete. IMPORTANT: Remember to refresh your materialized views in Supabase:")
            print("1. REFRESH MATERIALIZED VIEW public.weekly_sales_actuals;")
            print("2. REFRESH MATERIALIZED VIEW public.weekly_baselines;")
        else:
            print("No new sales data generated.")
    else:
        print("Data generation cancelled.")