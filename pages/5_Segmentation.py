# --- pages/4_🏬_Segmentation.py ---

import streamlit as st
import pandas as pd
from supabase import create_client, Client
from datetime import datetime
import holidays
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# --- Page Config ---
st.set_page_config(page_title="Retailer Segmentation", layout="wide")
st.title("🏬 Retailer Segmentation (Clustering)")
st.write("Using multiple unsupervised ML models to discover rich, multi-layered segments in retailer data.")
st.markdown("---")

# --- SHARED CODE: Load Data & Supabase Connection ---
@st.cache_data
def load_data():
    with st.spinner('Loading complete sales summary from Supabase...'):
        products = supabase.table("product_master").select("*").execute().data
        df_products = pd.DataFrame(products)
        all_summary_rows = []
        page = 0
        page_size = 1000
        while True:
            range_start = page * page_size
            range_end = range_start + page_size - 1
            page_data = supabase.table("daily_sales_summary").select("*").range(range_start, range_end).execute().data
            all_summary_rows.extend(page_data)
            if len(page_data) < page_size: break
            page += 1
        df_sales = pd.DataFrame(all_summary_rows)
    df_sales.rename(columns={'sale_date': 'created_at','total_units_sold': 'units_sold','average_sale_price': 'average_sale_price','was_on_promotion': 'on_promotion'}, inplace=True)
    df_sales["created_at"] = pd.to_datetime(df_sales["created_at"], utc=True)
    today = pd.to_datetime('today').tz_localize('UTC')
    df_sales = df_sales[df_sales['created_at'] <= today].copy()
    return df_products, df_sales

try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Configuration error: {e}. Please check your secrets.", icon="🚨")
    st.stop()

df_products, df_sales = load_data()
store_master = supabase.table("store_master").select("*").execute().data
df_stores = pd.DataFrame(store_master)
# --- END SHARED CODE ---

# --- 1. Gold Standard Data Preparation ---
st.subheader("1. Store Performance Metrics")
df_stores['channel_region'] = df_stores['store_channel'] + '_' + df_stores['store_region']
df_sales['revenue'] = df_sales['units_sold'] * df_sales['average_sale_price']
df_sales_merged = df_sales.merge(df_products[['product_id', 'product_category', 'product_era']], on='product_id', how='left')
# ... (The rest of your full data prep logic creating the 'store_features' DataFrame) ...
st.dataframe(store_features) # Display the final feature table

# --- Initialize session state ---
if 'segmented_df' not in st.session_state:
    st.session_state.segmented_df = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False


# --- 2. Multi-Lens Segmentation ---
st.markdown("---")
st.subheader("2. Run Multi-Lens Segmentation Models")
st.write("Select the number of clusters for each strategic lens and run the analysis.")

col1, col2, col3 = st.columns(3)
with col1:
    k_trend = st.number_input("K for Trend Adoption:", min_value=2, max_value=8, value=3, key="k_trend")
with col2:
    k_value = st.number_input("K for Business Value:", min_value=2, max_value=8, value=3, key="k_value")
with col3:
    k_mix = st.number_input("K for Product Mix:", min_value=2, max_value=8, value=3, key="k_mix")

if st.button("Run All Segmentation Models"):
    with st.spinner("Running all segmentation models..."):
        # --- Model 1: Trend Adoption ---
        trend_features = store_features[['overall_new_era_share', 'new_era_momentum']]
        scaler_trend = StandardScaler()
        scaled_features_trend = scaler_trend.fit_transform(trend_features)
        kmeans_trend = KMeans(n_clusters=k_trend, n_init='auto', random_state=42)
        store_features['trend_segment'] = kmeans_trend.fit_predict(scaled_features_trend)
        
        # --- Model 2: Business Value ---
        value_features = store_features[['total_revenue', 'revenue_growth_qoq']]
        scaler_value = StandardScaler()
        scaled_features_value = scaler_value.fit_transform(value_features)
        kmeans_value = KMeans(n_clusters=k_value, n_init='auto', random_state=42)
        store_features['value_segment'] = kmeans_value.fit_predict(scaled_features_value)

        # --- Model 3: Product Mix ---
        mix_features = store_features[['unique_products', 'overall_pct_chocolate', 'overall_pct_candy', 'overall_pct_ice_cream']]
        scaler_mix = StandardScaler()
        scaled_features_mix = scaler_mix.fit_transform(mix_features)
        kmeans_mix = KMeans(n_clusters=k_mix, n_init='auto', random_state=42)
        store_features['mix_segment'] = kmeans_mix.fit_predict(scaled_features_mix)
        
        st.session_state.segmented_df = store_features
        st.session_state.analysis_done = False # Reset AI analysis
        st.success("All three segmentations are complete!")

# --- 3. Display Results ---
if st.session_state.segmented_df is not None:
    segmented_df = st.session_state.segmented_df
    st.subheader("3. Segmentation Results")
    
    # Visualizations in tabs
    tab_trend, tab_value, tab_mix = st.tabs(["Trend Adoption Segments", "Value Segments", "Product Mix Segments"])
    
    with tab_trend:
        fig_trend = px.scatter(segmented_df, x='overall_new_era_share', y='new_era_momentum', color='trend_segment', hover_name='store_name')
        st.plotly_chart(fig_trend, use_container_width=True)
    with tab_value:
        fig_value = px.scatter(segmented_df, x='total_revenue', y='revenue_growth_qoq', color='value_segment', hover_name='store_name')
        st.plotly_chart(fig_value, use_container_width=True)
    with tab_mix:
        fig_mix = px.scatter(segmented_df, x='overall_pct_chocolate',
