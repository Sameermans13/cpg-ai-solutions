# pages/5_📈_Promotion_Analytics.py

import streamlit as st
import pandas as pd
from supabase import create_client, Client
import plotly.express as px
from datetime import datetime

# --- Supabase Connection & Data Fetching ---
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_supabase()

@st.cache_data(ttl=3600)
def fetch_data(table_name):
    try:
        response = supabase.table(table_name).select("*").execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_all_promotion_data():
    """
    Fetches and joins all promotion-related data with explicit column selections 
    to prevent merge conflicts and ensure a clean, final DataFrame.
    """
    # 1. Fetch all raw data from Supabase
    promotions_df = fetch_data("promotions_master")
    promo_products_df = fetch_data("promotion_products")
    promo_stores_df = fetch_data("promotion_stores")
    products_df = fetch_data("product_master")
    stores_df = fetch_data("store_master")

    # 2. Basic validation
    if promotions_df.empty or promo_products_df.empty:
        st.warning("Promotion master or product linking tables are empty.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 3. --- CORE FIX: Explicitly select columns BEFORE merging ---
    # This prevents Pandas from creating suffixed columns like 'column_x' and 'column_y'
    
    # Select only the promotion-level columns you need from the master table
    promotions_subset_df = promotions_df[[
        'promotion_id', 'promotion_name', 'promotion_tactic', 
        'start_date', 'end_date'
    ]].copy()

    # Select the product-specific promotional metrics
    promo_products_selected_df = promo_products_df[[
        'promotion_id', 'product_id', 'projected_unit_lift', 
        'planned_baseline_units', 'planned_product_spend', 'discount_percentage'
    ]].copy()

    # 4. Perform sequential merges with clean, pre-selected DataFrames
    
    # Merge promotions with the specific products in them
    promo_details_df = promotions_subset_df.merge(
        promo_products_selected_df, on='promotion_id', how='left'
    )
    
    # Merge with product master to get product details
    promo_details_df = promo_details_df.merge(
        products_df[['product_id', 'product_name', 'retail_price', 'margin_percentage']], 
        on='product_id', 
        how='left'
    )
    
    # Merge with promo_stores to link stores to the promotion
    promo_details_df = promo_details_df.merge(
        promo_stores_df[['promotion_id', 'store_id']], 
        on='promotion_id', 
        how='left'
    )
    
    # Merge with store_master to get store names
    promo_details_df = promo_details_df.merge(
        stores_df[['store_id', 'store_name']], 
        on='store_id', 
        how='left'
    )
    
    # 5. Final data cleaning and type casting for robustness
    
    # Ensure date columns are datetime objects
    promo_details_df['start_date'] = pd.to_datetime(promo_details_df['start_date'])
    promo_details_df['end_date'] = pd.to_datetime(promo_details_df['end_date'])
    

    # --- FIX: Make datetimes timezone-naive ---
    promo_details_df['start_date'] = promo_details_df['start_date'].dt.tz_localize(None)
    promo_details_df['end_date'] = promo_details_df['end_date'].dt.tz_localize(None)
    # --- END OF FIX ---


    # Drop rows where the merge might have failed to find a product or store
    promo_details_df.dropna(subset=['product_id', 'store_id'], inplace=True)
    
    # Use Pandas' nullable integer type for IDs to handle potential NaNs gracefully
    # and ensure accurate matching with other tables.
    promo_details_df['product_id'] = promo_details_df['product_id'].astype(pd.Int64Dtype())
    promo_details_df['store_id'] = promo_details_df['store_id'].astype(pd.Int64Dtype())

    return promo_details_df, products_df, stores_df


@st.cache_data(ttl=3600)
def fetch_weekly_actuals_with_baselines():
    weekly_actuals_df = fetch_data("weekly_sales_actuals")
    baselines_df = fetch_data("weekly_baselines")

    if weekly_actuals_df.empty:
        st.warning("weekly_sales_actuals is empty")
        return pd.DataFrame()
    
    # --- Make all date and ID conversions first ---
    weekly_actuals_df['week_start_date'] = pd.to_datetime(weekly_actuals_df['week_start_date'], errors='coerce').dt.tz_localize(None)
    weekly_actuals_df['product_id'] = pd.to_numeric(weekly_actuals_df['product_id'], errors='coerce').astype(pd.Int64Dtype())
    weekly_actuals_df['store_id'] = pd.to_numeric(weekly_actuals_df['store_id'], errors='coerce').astype(pd.Int64Dtype())
    
    if baselines_df.empty:
        st.warning("weekly_baselines is empty")
        weekly_actuals_df['baseline_avg_units'] = 0
        return weekly_actuals_df

    baselines_df['product_id'] = pd.to_numeric(baselines_df['product_id'], errors='coerce').astype(pd.Int64Dtype())
    baselines_df['store_id'] = pd.to_numeric(baselines_df['store_id'], errors='coerce').astype(pd.Int64Dtype())

    weekly_actuals_df['week_of_year'] = weekly_actuals_df['week_start_date'].dt.isocalendar().week.astype(int)

    # Merge actuals and baselines
    merged_df = weekly_actuals_df.merge(
        baselines_df,
        on=['product_id', 'store_id', 'week_of_year'],
        how='left'
    )

    # Fill NaNs after the merge
    merged_df['baseline_avg_units'] = pd.to_numeric(merged_df.get('baseline_avg_units', 0), errors='coerce').fillna(0)
    merged_df['units_sold'] = pd.to_numeric(merged_df.get('units_sold', 0), errors='coerce').fillna(0)
    merged_df['net_revenue'] = pd.to_numeric(merged_df.get('net_revenue', 0), errors='coerce').fillna(0)
    
    return merged_df

# --- Gold Standard KPI Calculation Engine ---
def calculate_promo_kpis(promo_details_df, weekly_data_df):
    """
    Calculates promotional KPIs using a vectorized approach for efficiency and accuracy.
    """
    if promo_details_df.empty:
        return pd.DataFrame()
        
    # 1. Filter the weekly sales data to only include dates relevant to ANY promotion
    min_date = promo_details_df['start_date'].min()
    max_date = promo_details_df['end_date'].max()
    
    promo_period_sales = weekly_data_df[
        (weekly_data_df['week_start_date'] >= min_date) & 
        (weekly_data_df['week_start_date'] <= max_date)
    ].copy()

    # 2. Join the promotion details with the relevant weekly sales
    kpis_df = promo_details_df.merge(
        promo_period_sales,
        on=['product_id', 'store_id'],
        how='left'
    )
    
    # 3. Filter for sales within the specific promotion's date range
    # First, fill NaT for non-matching rows to avoid errors
    kpis_df['week_start_date'] = pd.to_datetime(kpis_df['week_start_date'])
    
    # Create a boolean mask for valid sales dates
    mask = (
        (kpis_df['week_start_date'] >= kpis_df['start_date']) &
        (kpis_df['week_start_date'] <= kpis_df['end_date'])
    )
    # Get all rows that were part of the original promotion details
    # This ensures we keep promotions that had zero sales
    original_promo_rows = promo_details_df.set_index(
        ['promotion_id', 'product_id', 'store_id']
    ).index
    
    kpis_df_indexed = kpis_df.set_index(['promotion_id', 'product_id', 'store_id'])
    
    # Apply the mask, but also keep all original promo rows
    final_df = kpis_df[mask | kpis_df_indexed.index.isin(original_promo_rows)]

    # 4. Group by the original promotion details to aggregate the sales
    agg_cols = {
        'units_sold': 'sum',
        'net_revenue': 'sum',
        'baseline_avg_units': 'sum'
    }
    
    grouping_keys = [
        'promotion_id', 'promotion_name', 'product_id', 'product_name', 'store_id', 'store_name',
        'promotion_tactic', 'start_date', 'end_date', 'retail_price', 'margin_percentage',
        'projected_unit_lift', 'planned_baseline_units', 'discount_percentage',
        'planned_product_spend'
    ]
    
    promo_level_kpis = final_df.groupby(grouping_keys, dropna=False).agg(agg_cols).reset_index()

    # Rename aggregated columns for clarity
    promo_level_kpis.rename(columns={
        'units_sold': 'actual_units',
        'net_revenue': 'actual_revenue',
        'baseline_avg_units': 'baseline_units'
    }, inplace=True)
    
    # --- 5. Perform KPI Calculations on the entire DataFrame at once (Vectorized) ---
    df = promo_level_kpis 
    
    df['avg_price'] = df['actual_revenue'] / df['actual_units']
    df['avg_price'] = df['avg_price'].fillna(df['retail_price'] * (1 - df['discount_percentage']))

    # Planned Metrics
    df['planned_baseline_revenue'] = df['planned_baseline_units'] * df['avg_price']
    df['planned_lift_units'] = df['planned_baseline_units'] * df['projected_unit_lift']
    df['projected_total_units'] = df['planned_baseline_units'] + df['planned_lift_units']
    df['projected_revenue'] = df['projected_total_units'] * df['avg_price']
    df['planned_revenue_increase'] = df['projected_revenue'] - df['planned_baseline_revenue']
    df['cost_of_goods_planned'] = df['planned_revenue_increase'] * (1 - df['margin_percentage'])
    df['incremental_gross_margin_plan'] = df['planned_revenue_increase'] - df['cost_of_goods_planned']
    df['net_plan_margin'] = df['incremental_gross_margin_plan'] - df['planned_product_spend']
    df['projected_roi'] = df['net_plan_margin'] / df['planned_product_spend']

    # Actual Metrics
    df['actual_spend'] = df['planned_product_spend']
    df['baseline_revenue_actual'] = df['baseline_units'] * df['avg_price']
    df['incremental_revenue'] = df['actual_revenue'] - df['baseline_revenue_actual']
        # --- ADD THIS NEW LINE ---
    df['actual_cost_of_goods'] = df['incremental_revenue'] * (1 - df['margin_percentage'])
    # --- END OF ADDITION ---

    df['incremental_margin'] = df['incremental_revenue'] * df['margin_percentage']
    df['bottom_line_impact'] = df['incremental_margin'] - df['actual_spend']
    df['actual_roi'] = df['bottom_line_impact'] / df['actual_spend']
    
    df.fillna({'projected_roi': 0, 'actual_roi': 0}, inplace=True)
    df.replace([float('inf'), -float('inf')], {'projected_roi': 0, 'actual_roi': 0}, inplace=True)
    
    # --- *** FINAL FIX: Rename column to match UI expectation *** ---
    df.rename(columns={'planned_product_spend': 'planned_spend'}, inplace=True)
    
    return df


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Promotion Analytics")
st.title("📈 Promotion Effectiveness: Margin & ROI Analysis")

# --- This part remains as you have it ---
promo_details_df, _, _ = fetch_all_promotion_data()
weekly_data_df = fetch_weekly_actuals_with_baselines()

if not promo_details_df.empty and not weekly_data_df.empty:
    st.sidebar.header("Filter Promotions")
    promo_names = ["All"] + sorted(promo_details_df['promotion_name'].unique().tolist())
    selected_promo = st.sidebar.selectbox("Select Promotion", promo_names)
    
    filtered_promos = promo_details_df if selected_promo == "All" else promo_details_df[promo_details_df['promotion_name'] == selected_promo]
    
    # Calculate the detailed, product-store-level KPIs using your existing function
    kpis_df = calculate_promo_kpis(filtered_promos, weekly_data_df)

    if not kpis_df.empty:
        # --- Create an Aggregated, Promo-Level Summary DataFrame ---
        agg_cols = {
            'planned_baseline_units': 'sum', 'planned_baseline_revenue': 'sum',
            'planned_lift_units': 'sum', 'projected_total_units': 'sum',
            'projected_revenue': 'sum', 'planned_revenue_increase': 'sum',
            'planned_spend': 'sum', 'cost_of_goods_planned': 'sum',
            'incremental_gross_margin_plan': 'sum', 'net_plan_margin': 'sum',
            'actual_units': 'sum', 'actual_revenue': 'sum',
            'actual_spend': 'sum', 'incremental_revenue': 'sum',
            'actual_cost_of_goods': 'sum', 'incremental_margin': 'sum',
            'bottom_line_impact': 'sum'
        }
        
        promo_summary_df = kpis_df.groupby('promotion_name').agg(agg_cols).reset_index()

        # Re-calculate ROI on the aggregated summary to be accurate
        promo_summary_df['projected_roi'] = promo_summary_df['net_plan_margin'] / promo_summary_df['planned_spend']
        promo_summary_df['actual_roi'] = promo_summary_df['bottom_line_impact'] / promo_summary_df['actual_spend']
        promo_summary_df.replace([float('inf'), -float('inf')], 0, inplace=True)
        promo_summary_df.fillna(0, inplace=True)
        
        # --- Define Formatting for all tables ---
        formatter = {
            "Baseline Units": "{:,.0f}", "Incremental Units": "{:,.0f}", "Total Units": "{:,.0f}",
            "Planned Baseline Units": "{:,.0f}", "Planned Total Units": "{:,.0f}",
            "Actual Total Units": "{:,.0f}",
            "Baseline Revenue": "${:,.0f}", "Total Revenue": "${:,.0f}",
            "Total Incremental Revenue": "${:,.0f}", "Total Spend": "${:,.0f}",
            "Planned Incremental Revenue": "${:,.0f}", "Cost of Goods": "${:,.0f}",
            "Total Gross Incremental Margin": "${:,.0f}", "Actual Incremental Revenue": "${:,.0f}",
            "Actual Cost of Goods": "${:,.0f}", "Actual Promo Spend": "${:,.0f}",
            "Actual Incremental Gross Margin": "${:,.0f}",
            "Projected ROI": "{:.1%}", "Actual ROI": "{:.1%}"
        }

        # --- Display Table 1: Promotion Plan ---
        st.subheader("1. Promotion Plan")
        plan_summary_cols = {
            'promotion_name': 'Promotion',
            'planned_baseline_units': 'Baseline Units',
            'planned_baseline_revenue': 'Baseline Revenue',
            'planned_lift_units': 'Incremental Units',
            'projected_total_units': 'Total Units',
            'projected_revenue': 'Total Revenue',
            'planned_revenue_increase': 'Total Incremental Revenue',
            'planned_spend': 'Total Spend'
        }
        st.dataframe(promo_summary_df[plan_summary_cols.keys()].rename(columns=plan_summary_cols).style.format(formatter), use_container_width=True)

        with st.expander("View Plan Details (by Product & Store)"):
            plan_detail_cols = { 'product_name': 'Product', 'store_name': 'Store', **plan_summary_cols }
            del plan_detail_cols['promotion_name']
            st.dataframe(kpis_df[plan_detail_cols.keys()].rename(columns=plan_detail_cols).style.format(formatter), use_container_width=True)

        # --- Display Table 2: Projected ROI ---
        st.subheader("2. Projected ROI")
        proj_roi_summary_cols = {
            'promotion_name': 'Promotion',
            'planned_revenue_increase': 'Planned Incremental Revenue',
            'cost_of_goods_planned': 'Cost of Goods',
            'planned_spend': 'Total Spend',
            'incremental_gross_margin_plan': 'Total Gross Incremental Margin',
            'projected_roi': 'Projected ROI'
        }
        st.dataframe(promo_summary_df[proj_roi_summary_cols.keys()].rename(columns=proj_roi_summary_cols).style.format(formatter), use_container_width=True)

        with st.expander("View Projected ROI Details (by Product & Store)"):
            proj_roi_detail_cols = { 'product_name': 'Product', 'store_name': 'Store', **proj_roi_summary_cols }
            del proj_roi_detail_cols['promotion_name']
            st.dataframe(kpis_df[proj_roi_detail_cols.keys()].rename(columns=proj_roi_detail_cols).style.format(formatter), use_container_width=True)
            
        # --- Display Table 3: Plan vs Actual Sales ---
        st.subheader("3. Plan vs. Actual Sales (Units)")
        plan_vs_actual_summary_cols = {
            'promotion_name': 'Promotion',
            'planned_baseline_units': 'Planned Baseline Units',
            'projected_total_units': 'Planned Total Units',
            'actual_units': 'Actual Total Units'
        }
        st.dataframe(promo_summary_df[plan_vs_actual_summary_cols.keys()].rename(columns=plan_vs_actual_summary_cols).style.format(formatter), use_container_width=True)

        with st.expander("View Sales Details (by Product & Store)"):
            plan_vs_actual_detail_cols = { 'product_name': 'Product', 'store_name': 'Store', **plan_vs_actual_summary_cols }
            del plan_vs_actual_detail_cols['promotion_name']
            st.dataframe(kpis_df[plan_vs_actual_detail_cols.keys()].rename(columns=plan_vs_actual_detail_cols).style.format(formatter), use_container_width=True)

        # --- Display Table 4: Plan vs Actual ROI ---
        st.subheader("4. Plan vs. Actual ROI")
        roi_vs_actual_summary_cols = {
            'promotion_name': 'Promotion',
            'planned_revenue_increase': 'Planned Incremental Revenue',
            'incremental_revenue': 'Actual Incremental Revenue',
            'actual_cost_of_goods': 'Actual Cost of Goods',
            'actual_spend': 'Actual Promo Spend',
            'incremental_margin': 'Actual Incremental Gross Margin',
            'actual_roi': 'Actual ROI'
        }
        st.dataframe(promo_summary_df[roi_vs_actual_summary_cols.keys()].rename(columns=roi_vs_actual_summary_cols).style.format(formatter), use_container_width=True)
        
        with st.expander("View ROI Details (by Product & Store)"):
            roi_vs_actual_detail_cols = { 'product_name': 'Product', 'store_name': 'Store', **roi_vs_actual_summary_cols }
            del roi_vs_actual_detail_cols['promotion_name']
            st.dataframe(kpis_df[roi_vs_actual_detail_cols.keys()].rename(columns=roi_vs_actual_detail_cols).style.format(formatter), use_container_width=True)