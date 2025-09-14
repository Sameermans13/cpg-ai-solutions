import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from datetime import datetime
import holidays

# --------------------------
# Supabase Connection
# --------------------------
# It's best practice to get these from secrets, even for the anon key.
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
except KeyError:
    st.error("Supabase URL or Key not found in secrets. Please check your .streamlit/secrets.toml file.")
    st.stop()


# --------------------------
# Fetch Data Function
# --------------------------
@st.cache_data
def load_data():
    """
    This function correctly fetches ALL data from the materialized view,
    handling pagination to overcome the 1,000-row limit.
    """
    with st.spinner('Loading complete sales summary from Supabase...'):
        # Fetch products (small table, no pagination needed)
        products = supabase.table("product_master").select("*").execute().data
        df_products = pd.DataFrame(products)

        # --- Paginated fetch for the materialized view ---
        all_summary_rows = []
        page = 0
        page_size = 1000
        while True:
            range_start = page * page_size
            range_end = range_start + page_size - 1
            page_data = supabase.table("daily_sales_summary").select("*").range(range_start, range_end).execute().data
            
            all_summary_rows.extend(page_data)
            
            if len(page_data) < page_size:
                break
            page += 1
            
        df_sales = pd.DataFrame(all_summary_rows)
        
    # --- Data Cleaning and Feature Engineering ---
    df_sales.rename(columns={
        'sale_date': 'created_at',
        'total_units_sold': 'units_sold',
        'average_sale_price': 'average_sale_price',
        'was_on_promotion': 'on_promotion'
    }, inplace=True)

    df_sales["created_at"] = pd.to_datetime(df_sales["created_at"], utc=True)
    
    # Filter out future data
    today = pd.to_datetime('today').tz_localize('UTC')
    df_sales = df_sales[df_sales['created_at'] <= today].copy()

    # Feature engineering logic
    df_sales['on_promotion'] = df_sales['on_promotion'].fillna(False).astype(bool)
    df_sales['day_of_week'] = df_sales['created_at'].dt.day_name()
    df_sales['month'] = df_sales['created_at'].dt.month
    df_sales['week_of_year'] = df_sales['created_at'].dt.isocalendar().week
    us_holidays = holidays.US(years=df_sales['created_at'].dt.year.unique())
    df_sales['is_holiday'] = df_sales['created_at'].dt.date.isin(us_holidays)

    return df_products, df_sales

# --- Load the data ---
df_products, df_sales = load_data()




# --------------------------
# Dashboard Layout
# --------------------------
st.set_page_config(page_title="CPG Sales Dashboard", layout="wide")
st.title("📊 CPG Sales Dashboard")
st.write("CPG product sales analytics (using demo dataset)")

# Sidebar Filters
st.sidebar.header("Filters")
category_filter = st.sidebar.multiselect(
    "Select Product Category", 
    options=df_products["product_category"].unique()
)
date_range = st.sidebar.date_input(
    "Select Date Range", 
    [df_sales["created_at"].min(), df_sales["created_at"].max()]
)

# Apply Filters
df_filtered = df_sales.copy()
if category_filter:
    df_filtered = df_filtered[
        df_filtered["product_id"].isin(
            df_products[df_products["product_category"].isin(category_filter)]["product_id"]
        )
    ]
if date_range:
    start_date, end_date = date_range
    df_filtered = df_filtered[
        (df_filtered["created_at"].dt.date >= start_date) &
        (df_filtered["created_at"].dt.date <= end_date)
    ]

# Before merging, drop the unnecessary 'created_at' column from the products dataframe
# to avoid column name collisions.
if 'created_at' in df_products.columns:
    df_products_to_merge = df_products.drop(columns=['created_at'])
else:
    df_products_to_merge = df_products


# Merge product info
df_merged = df_filtered.merge(df_products_to_merge, on="product_id", how="left")




# --------------------------
# Revenue & KPI Metrics
# --------------------------
# The 'average_sale_price' column is now already in df_merged, so we can calculate directly.

# Calculate both Gross (potential) and Net (actual) Revenue
df_merged['gross_revenue'] = df_merged['units_sold'] * df_merged['retail_price']
df_merged['net_revenue'] = df_merged['units_sold'] * df_merged['average_sale_price']

# --- KPI Metrics ---
total_units = int(df_merged["units_sold"].sum())
total_net_revenue = float(df_merged["net_revenue"].sum())
total_gross_revenue = float(df_merged["gross_revenue"].sum()) if 'gross_revenue' in df_merged and df_merged["gross_revenue"].sum() > 0 else 0
discount_impact = total_gross_revenue - total_net_revenue

col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Total Units Sold", f"{total_units:,}")
col2.metric("💰 Net Revenue", f"${total_net_revenue:,.2f}", help="Actual revenue after all discounts.")
col3.metric("💲 Discount Impact", f"${discount_impact:,.2f}", help="The value of all promotions and discounts.")

# Avoid division by zero if there is no gross revenue
if total_gross_revenue > 0:
    price_realization = total_net_revenue / total_gross_revenue
    col4.metric("📈 Price Realization", f"{price_realization:.2%}", help="The percentage of potential revenue that was actually captured.")
else:
    col4.metric("📈 Price Realization", "N/A", help="The percentage of potential revenue that was actually captured.")

# --------------------------
# Sales by Day of Week
# --------------------------
st.subheader("Weekly Sales Performance")

# DEFENSIVE CHECK: Only run if there is data
if not df_merged.empty:
    sales_by_day = df_merged.groupby("day_of_week").agg(
        {"units_sold": "sum", "net_revenue": "sum", "gross_revenue": "sum"}
    ).reset_index()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sales_by_day['day_of_week'] = pd.Categorical(sales_by_day['day_of_week'], categories=day_order, ordered=True)
    sales_by_day = sales_by_day.sort_values('day_of_week')

    tab_day_net, tab_day_gross = st.tabs(["By Net Revenue", "By Gross Revenue"])

    with tab_day_net:
        fig_day_of_week_net = px.bar(
            sales_by_day, x="day_of_week", y="net_revenue", title="Net Revenue by Day of Week"
        )
        st.plotly_chart(fig_day_of_week_net, use_container_width=True)
    
    with tab_day_gross:
        fig_day_of_week_gross = px.bar(
            sales_by_day, x="day_of_week", y="gross_revenue", title="Gross Revenue by Day of Week"
        )
        st.plotly_chart(fig_day_of_week_gross, use_container_width=True)
else:
    st.warning("No data available for the selected filters.")



# --------------------------
# Promotion Impact Analysis
# --------------------------
st.subheader("Impact of Promotions")

if not df_merged.empty and 'on_promotion' in df_merged.columns:
    
    # Fill any potential nulls just in case, and ensure it's a boolean type
    df_merged['on_promotion'] = df_merged['on_promotion'].fillna(False).astype(bool)

    promo_impact = df_merged.groupby("on_promotion").agg(
        avg_units_sold=("units_sold", "mean")
    ).reset_index()

    # Check if we have both promotional and non-promotional data to compare
    if promo_impact['on_promotion'].nunique() > 1:
        promo_impact["on_promotion"] = promo_impact["on_promotion"].map({True: "On Promotion", False: "No Promotion"})

        fig_promo_impact = px.bar(
            promo_impact,
            x="on_promotion",
            y="avg_units_sold",
            title="Average Units Sold: Promotion vs. No Promotion",
            color="on_promotion",
            labels={"avg_units_sold": "Average Units Sold per Transaction"}
        )
        st.plotly_chart(fig_promo_impact, use_container_width=True)
    else:
        st.info("The current filter selection does not contain both promotional and non-promotional sales to compare.")
else:
    st.warning("No data available for the selected filters.")


# --------------------------
# Sales by Category (Gross vs. Net)
# --------------------------
st.subheader("Sales by Category")

sales_summary = df_merged.groupby("product_category").agg(
    {"units_sold": "sum", "gross_revenue": "sum", "net_revenue": "sum"}
).reset_index()

tab1, tab2 = st.tabs(["Net Revenue", "Gross Revenue"])

with tab1:
    fig_net_rev = px.bar(
        sales_summary, 
        x="product_category", 
        y="net_revenue", 
        text_auto='.2s',
        title="Net Revenue (Actual) by Category",
        color="product_category"
    )
    st.plotly_chart(fig_net_rev, use_container_width=True)

with tab2:
    fig_gross_rev = px.bar(
        sales_summary, 
        x="product_category", 
        y="gross_revenue", 
        text_auto='.2s',
        title="Gross Revenue (Potential) by Category",
        color="product_category"
    )
    st.plotly_chart(fig_gross_rev, use_container_width=True)



# --------------------------
# Trend Over Time
# --------------------------
st.subheader("Sales Trend Over Time")

if not df_merged.empty:
    # We now need to aggregate on the date from the merged dataframe
    sales_trend = df_merged.groupby(df_merged["created_at"].dt.date).agg(
        {"units_sold": "sum", "net_revenue": "sum", "gross_revenue": "sum"}
    ).reset_index()

    tab_trend_units, tab_trend_net, tab_trend_gross = st.tabs(["Units Sold Trend", "Net Revenue Trend", "Gross Revenue Trend"])

    with tab_trend_units:
        fig_trend_units = px.line(sales_trend, x="created_at", y="units_sold", markers=True, title="Units Sold Over Time")
        st.plotly_chart(fig_trend_units, use_container_width=True)

    with tab_trend_net:
        fig_trend_net = px.line(sales_trend, x="created_at", y="net_revenue", markers=True, title="Net Revenue Over Time")
        st.plotly_chart(fig_trend_net, use_container_width=True)

    with tab_trend_gross:
        fig_trend_gross = px.line(sales_trend, x="created_at", y="gross_revenue", markers=True, title="Gross Revenue Over Time")
        st.plotly_chart(fig_trend_gross, use_container_width=True)
else:
    st.warning("No data available for the selected filters.")


# --------------------------
# Top 5 Products
# --------------------------
st.subheader("Top 5 Products")

if not df_merged.empty:
    top_products_units = df_merged.groupby("product_name")["units_sold"].sum().nlargest(5).reset_index()
    top_products_net_rev = df_merged.groupby("product_name")["net_revenue"].sum().nlargest(5).reset_index()
    top_products_gross_rev = df_merged.groupby("product_name")["gross_revenue"].sum().nlargest(5).reset_index()

    tab_top_units, tab_top_net, tab_top_gross = st.tabs(["By Units Sold", "By Net Revenue", "By Gross Revenue"])

    with tab_top_units:
        fig_top_units = px.bar(top_products_units, x="product_name", y="units_sold", title="Top 5 Products by Units")
        st.plotly_chart(fig_top_units, use_container_width=True)

    with tab_top_net:
        fig_top_net_rev = px.bar(top_products_net_rev, x="product_name", y="net_revenue", title="Top 5 Products by Net Revenue")
        st.plotly_chart(fig_top_net_rev, use_container_width=True)

    with tab_top_gross:
        fig_top_gross_rev = px.bar(top_products_gross_rev, x="product_name", y="gross_revenue", title="Top 5 Products by Gross Revenue")
        st.plotly_chart(fig_top_gross_rev, use_container_width=True)
else:
    st.warning("No data available for the selected filters.")

