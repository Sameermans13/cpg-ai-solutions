# day2_cpg_analytics.py
from supabase import create_client, Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser as dp 

url = "https://sywxhehahunevputgxdd.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"

supabase = create_client(url, key)

# --- 2. Helper function to fetch table into DataFrame ---
def fetch_table(table_name, columns="*"):
    resp = supabase.table(table_name).select(columns).execute()
    data = resp.data
    if not data:
        return pd.DataFrame()
    return pd.DataFrame.from_records(data)

# --- 3. Load product_master and sales_transactions ---
df_products = fetch_table(
    "product_master",
    "product_id,product_name,product_category,product_sub_category,retail_price,created_at"
)

df_sales = fetch_table(
    "sales_transactions",
    "transaction_id,created_at,product_id,units_sold,sale_price,store_id,region"
)

print("Products rows:", len(df_products))
print("Sales rows:", len(df_sales))

# --- 4. Robust datetime parsing function ---
def robust_parse_series(s):
    parsed = pd.to_datetime(s.astype(str), utc=True, errors='coerce', infer_datetime_format=True)
    if parsed.isna().any():
        mask = parsed.isna()
        def parse_one(val):
            try:
                return pd.to_datetime(dp.isoparse(str(val)), utc=True)
            except Exception:
                try:
                    return pd.to_datetime(dp.parse(str(val)), utc=True)
                except Exception:
                    return pd.NaT
        parsed.loc[mask] = s[mask].apply(parse_one)
    return parsed

# --- 5. Apply datetime parsing to both tables ---
df_sales["created_at"] = robust_parse_series(df_sales["created_at"])
df_products["created_at"] = robust_parse_series(df_products["created_at"])

print("Parsed sales created_at dtype:", df_sales["created_at"].dtype)
print("Parsed products created_at dtype:", df_products["created_at"].dtype)

# -------------------------------
# 3. Clean & prepare columns / datetimes
# -------------------------------
if not df_sales.empty:
    # Parse timestamps (Supabase returns ISO strings)
    df_sales["created_at"] = pd.to_datetime(df_sales["created_at"], utc=True)
    # If you want local timezone, convert: df_sales["created_at"] = df_sales["created_at"].dt.tz_convert("US/Eastern")

if not df_products.empty:
    df_products["created_at"] = pd.to_datetime(df_products["created_at"], utc=True)

# -------------------------------
# 4. Join sales with product master (bring retail_price, product_name)
# -------------------------------
df = pd.merge(df_sales, df_products[["product_id", "product_name", "retail_price", "product_category"]], how="left", on="product_id")

# Create derived columns
df["revenue"] = df["units_sold"].astype(float) * df["sale_price"].astype(float)
# Flag if sale_price is below retail price (simple promo heuristic)
df["on_promo"] = df["sale_price"].astype(float) < df["retail_price"].astype(float)

# -------------------------------
# 5. Aggregations: product-level metrics
# -------------------------------
product_agg = df.groupby(["product_id", "product_name"]).agg(
    total_units_sold = ("units_sold", "sum"),
    total_revenue = ("revenue", "sum"),
    avg_sale_price = ("sale_price", "mean"),
    promo_pct = ("on_promo", lambda x: 100.0 * x.sum() / max(1, x.count()))
).reset_index().sort_values("total_revenue", ascending=False)

print("\nTop 10 products by revenue:\n", product_agg.head(10))

# -------------------------------
# 6. Aggregations: region-level & store-level
# -------------------------------
region_agg = df.groupby("region").agg(
    units_sold = ("units_sold", "sum"),
    revenue = ("revenue", "sum"),
    avg_price = ("sale_price", "mean")
).reset_index().sort_values("revenue", ascending=False)

print("\nRevenue by region:\n", region_agg)

# -------------------------------
# 7. Time series: daily sales & rolling average
# -------------------------------
# ensure index is datetime
df_ts = df.set_index("created_at").sort_index()

# daily units sold and revenue
daily = df_ts[["units_sold","revenue"]].resample("D").sum().fillna(0)
daily["units_7d_avg"] = daily["units_sold"].rolling(window=7, min_periods=1).mean()
daily["revenue_7d_avg"] = daily["revenue"].rolling(window=7, min_periods=1).mean()

print("\nDaily summary (last 10 rows):\n", daily.tail(10))

# -------------------------------
# 8. Pivot table: product x region revenue matrix
# -------------------------------
pivot = pd.pivot_table(df, values="revenue", index="product_name", columns="region", aggfunc="sum", fill_value=0)
print("\nPivot (product x region) - top rows:\n", pivot.head())

# -------------------------------
# 9. Basic charts with matplotlib
# -------------------------------
plt.figure(figsize=(10,6))
# Top 10 products by revenue bar chart
top10 = product_agg.head(10).set_index("product_name")
plt.bar(top10.index, top10["total_revenue"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Products by Revenue")
plt.ylabel("Total Revenue")
plt.tight_layout()
plt.show()

# Time series chart
plt.figure(figsize=(10,5))
plt.plot(daily.index, daily["revenue"], label="Daily Revenue")
plt.plot(daily.index, daily["revenue_7d_avg"], label="7-day Avg")
plt.legend()
plt.title("Daily Revenue and 7-day Average")
plt.tight_layout()
plt.show()



product_agg.set_index("product_id")[["total_units_sold", "total_revenue"]].plot(kind="bar")
plt.title("Product Performance (Quantity vs Revenue)")
plt.ylabel("Values")
plt.show()


# Sort by total revenue (descending)
top_products = product_agg.sort_values(by="total_revenue", ascending=False)
print(top_products.head(5))  # Top 5 products

# Add a rank column
top_products["rank_by_revenue"] = top_products["total_revenue"].rank(
    method="dense", ascending=False
).astype(int)

print(top_products.head(10))  # See top 10 with ranks

# Plot Top 5 products by revenue
top5 = top_products.head(5)

plt.bar(top5["product_id"], top5["total_revenue"])
plt.title("Top 5 Products by Revenue")
plt.ylabel("Revenue")
plt.show()

# Calculate contribution percentage
top_products["revenue_pct"] = (
    top_products["total_revenue"] / top_products["total_revenue"].sum() * 100
)

# Calculate cumulative contribution
top_products["cumulative_pct"] = top_products["revenue_pct"].cumsum()

print(top_products[["product_id", "total_revenue", "revenue_pct", "cumulative_pct"]])

# -------------------------------
# 11. Examples: push aggregated analytics back to Supabase (optional)
#    (Uncomment only if you have a product_analytics table and want to upsert)
# -------------------------------
# for _, row in product_agg.iterrows():
#     supabase.table("product_analytics").upsert({
#         "product_id": int(row["product_id"]),
#         "period_start": pd.Timestamp.today().date().isoformat(),
#         "period_end": pd.Timestamp.today().date().isoformat(),
#         "total_units_sold": int(row["total_units_sold"]),
#         "total_revenue": float(row["total_revenue"]),
#         "avg_price": float(row["avg_sale_price"]),
#         "on_promotion": float(row["promo_pct"]) > 50.0
#     }).execute()
#
# print("Upserted analytics to product_analytics (if configured)")