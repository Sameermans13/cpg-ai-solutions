# --- pages/4_🏬_Segmentation.py ---

import streamlit as st
import pandas as pd
from supabase import create_client, Client
from datetime import datetime
import holidays

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import numpy as np


# --- Page Config ---
st.set_page_config(page_title="Retailer Segmentation", layout="wide")
st.title("🏬 Retailer Segmentation (Clustering)")
st.write("Using unsupervised machine learning (K-Means) to discover hidden segments in retailer data.")
st.markdown("---")

# --- SHARED CODE: Load Data ---
@st.cache_data
def load_data():
    """
    This function correctly fetches ALL data from the materialized view,
    handling pagination to overcome the 1,000-row limit.
    """
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
            
            if len(page_data) < page_size:
                break
            page += 1
            
        df_sales = pd.DataFrame(all_summary_rows)
        
    df_sales.rename(columns={'sale_date': 'created_at','total_units_sold': 'units_sold','average_sale_price': 'average_sale_price','was_on_promotion': 'on_promotion'}, inplace=True)
    df_sales["created_at"] = pd.to_datetime(df_sales["created_at"], utc=True)
    today = pd.to_datetime('today').tz_localize('UTC')
    df_sales = df_sales[df_sales['created_at'] <= today].copy()
    df_sales['day_of_week'] = df_sales['created_at'].dt.day_name()
    return df_products, df_sales

# Supabase Connection
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
except Exception as e:
    st.error(f"Could not connect to Supabase. Please check your secrets. Error: {e}")
    st.stop()

df_products, df_sales = load_data()

# We also need the store master data for this page
store_master = supabase.table("store_master").select("*").execute().data
df_stores = pd.DataFrame(store_master)
# --- END SHARED CODE ---


# --- 1. Advanced Data Preparation with Trend & Geographic Features ---
st.subheader("1. Store Performance Metrics")
st.write("This profile for each store includes overall history, recent trends, and geographic attributes.")

# Create the geographic interaction feature in the stores dataframe
df_stores['channel_region'] = df_stores['store_channel'] + '_' + df_stores['store_region']

# --- Overall Historical Features ---
df_sales['revenue'] = df_sales['units_sold'] * df_sales['average_sale_price']

# --- THE FIX IS ON THIS LINE ---
df_sales_merged = df_sales.merge(df_products[['product_id', 'product_category']], on='product_id', how='left')
# --- END FIX ---

category_sales_overall = df_sales_merged.pivot_table(index='store_id', columns='product_category', values='units_sold', aggfunc='sum', fill_value=0)
category_sales_pct_overall = category_sales_overall.div(category_sales_overall.sum(axis=1), axis=0).fillna(0)
category_sales_pct_overall.columns = [f'overall_pct_{col.lower().replace(" ", "_")}' for col in category_sales_pct_overall.columns]

store_features = df_sales_merged.groupby('store_id').agg(
    total_revenue=('revenue', 'sum'),
    total_units=('units_sold', 'sum'),
    unique_products=('product_id', 'nunique'),
).reset_index()

store_features = store_features.merge(category_sales_pct_overall, on='store_id', how='left')

# --- Calculate Recent Performance & Trend Features ---
# Define time periods
last_date = df_sales_merged['created_at'].max()
recent_start_date = last_date - pd.Timedelta(days=90)
previous_start_date = last_date - pd.Timedelta(days=180)

# Filter for recent and previous sales periods
df_recent = df_sales_merged[df_sales_merged['created_at'] >= recent_start_date]
df_previous = df_sales_merged[(df_sales_merged['created_at'] >= previous_start_date) & (df_sales_merged['created_at'] < recent_start_date)]

# Calculate revenue for each period to find the growth trend
revenue_recent = df_recent.groupby('store_id')['revenue'].sum().reset_index().rename(columns={'revenue': 'revenue_last_90d'})
revenue_previous = df_previous.groupby('store_id')['revenue'].sum().reset_index().rename(columns={'revenue': 'revenue_prev_90d'})

# Calculate recent category mix to find the product mix trend
category_sales_recent = df_recent.pivot_table(index='store_id', columns='product_category', values='units_sold', aggfunc='sum', fill_value=0)
category_sales_pct_recent = category_sales_recent.div(category_sales_recent.sum(axis=1), axis=0).fillna(0)
category_sales_pct_recent.columns = [f'recent_pct_{col.lower().replace(" ", "_")}' for col in category_sales_pct_recent.columns]

# --- Merge Trend Data and Create Final Trend Features ---
store_features = store_features.merge(revenue_recent, on='store_id', how='left')
store_features = store_features.merge(revenue_previous, on='store_id', how='left')
store_features = store_features.merge(category_sales_pct_recent, on='store_id', how='left')
store_features.fillna(0, inplace=True) # Fill NaNs for stores with no recent/previous sales

# Calculate the final trend features
store_features['revenue_growth_qoq'] = (store_features['revenue_last_90d'] - store_features['revenue_prev_90d']) / (store_features['revenue_prev_90d'] + 1)
store_features['chocolate_mix_trend'] = store_features['recent_pct_chocolate'] - store_features['overall_pct_chocolate']
store_features['candy_mix_trend'] = store_features['recent_pct_candy'] - store_features['overall_pct_candy']
store_features['ice_cream_mix_trend'] = store_features['recent_pct_ice_cream'] - store_features['overall_pct_ice_cream']


# --- Final Merge with Store Details ---
# Merge with store names and our new channel_region feature
store_features = store_features.merge(df_stores[['store_id', 'store_name', 'store_channel', 'channel_region']], on='store_id', how='left')

st.dataframe(store_features)

st.info("This table is the input for our clustering model. Each row is a store, and each column is a feature describing that store's behavior.", icon="💡")


# --- 2. Feature Scaling & Elbow Method ---
st.markdown("---")
st.subheader("2. Finding the Optimal Number of Clusters (K)")
st.write("First, we must scale our features. Then, we use the 'Elbow Method' to find the ideal number of segments.")

# Select only the numerical features for the model
# We exclude IDs and names, which are not features for the model
features_for_clustering = store_features.select_dtypes(include=np.number).drop(columns=['store_id'])

# Check if there are features to process
if not features_for_clustering.empty:
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)

    # --- Elbow Method Logic ---
    inertia = []
    k_range = range(1, 11) # Test for K from 1 to 10 clusters

    with st.spinner("Calculating optimal number of clusters..."):
        for k in k_range:
            # Check if k is less than or equal to the number of samples
            if k <= len(store_features):
                kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
                kmeans.fit(scaled_features)
                inertia.append(kmeans.inertia_)

    # --- Plot the Elbow Curve ---
    if inertia:
        fig = go.Figure(data=go.Scatter(x=list(k_range), y=inertia, mode='lines+markers'))
        fig.update_layout(
            title='Elbow Method for Optimal K',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Inertia (Within-cluster sum of squares)'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("Look for the 'elbow' in the chart above – the point where the line starts to flatten out (often around K=3, 4, or 5). This point represents a good balance and is the 'K' we'll use in the next step.", icon="🎯")
    else:
        st.warning("Could not generate the Elbow Method plot. Not enough data points available for the tested range of clusters.")
else:
    st.warning("No data available to perform clustering.")




