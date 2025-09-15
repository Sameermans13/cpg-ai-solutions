import streamlit as st
import pandas as pd
from supabase import create_client, Client
from datetime import datetime
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
st.write("Using multiple ML models to discover rich, multi-layered segments in retailer data.")
st.markdown("---")

# --- Session State Initialization ---
# This ensures our variables persist across reruns
if 'scaled_features' not in st.session_state: st.session_state.scaled_features = None
if 'segmented_df' not in st.session_state: st.session_state.segmented_df = None

# --- SHARED CODE: Load Data & Supabase Connection ---
@st.cache_data
def load_data():
    # ... (Your complete, paginated load_data function) ...
    return df_products, df_sales

try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Configuration error: {e}. Please check your secrets.", icon="🚨")
    st.stop()

# --- 1. Data Preparation ---
@st.cache_data
def prepare_segmentation_data():
    df_products, df_sales = load_data()
    store_master = supabase.table("store_master").select("*").execute().data
    df_stores = pd.DataFrame(store_master)
    
    # ... (The full 'Gold Standard' data preparation logic goes here) ...
    # This creates the final 'store_features' DataFrame
    
    return store_features

store_features = prepare_segmentation_data()
st.subheader("1. Store Feature Profiles")
st.dataframe(store_features)


# --- 2. Feature Scaling & Elbow Method ---
st.markdown("---")
st.subheader("2. Finding the Optimal Number of Clusters (K)")
features_for_clustering = store_features.select_dtypes(include=np.number).drop(columns=['store_id'])
if not features_for_clustering.empty:
    scaler = StandardScaler()
    # Save scaled features to session state to be used by the next step
    st.session_state.scaled_features = scaler.fit_transform(features_for_clustering)
    
    inertia = []
    k_range = range(1, 11)
    with st.spinner("Calculating optimal clusters..."):
        for k in k_range:
            if k <= len(store_features):
                kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
                kmeans.fit(st.session_state.scaled_features)
                inertia.append(kmeans.inertia_)
    if inertia:
        fig = go.Figure(data=go.Scatter(x=list(k_range), y=inertia, mode='lines+markers'))
        fig.update_layout(title='Elbow Method for Optimal K', xaxis_title='Number of Clusters (K)', yaxis_title='Inertia')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Look for the 'elbow' in the chart to determine the best 'K'.", icon="🎯")

# --- 3. Final Segmentation and Analysis ---
st.markdown("---")
st.subheader("3. Final Segmentation and Analysis")

if st.session_state.scaled_features is not None:
    optimal_k = st.number_input("Based on the elbow chart, select K:", min_value=2, max_value=10, value=4, step=1)
    if st.button("Run Final Segmentation"):
        with st.spinner("Training final model..."):
            kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
            kmeans.fit(st.session_state.scaled_features)
            results_df = store_features.copy()
            results_df['cluster_original'] = kmeans.labels_
            cluster_profiles = results_df.groupby('cluster_original').mean(numeric_only=True)
            sorted_clusters = cluster_profiles.sort_values(by='total_revenue', ascending=False).reset_index()
            stable_cluster_map = {v: k for k, v in sorted_clusters['cluster_original'].to_dict().items()}
            results_df['cluster'] = results_df['cluster_original'].map(stable_cluster_map)
            st.session_state.segmented_df = results_df
            st.success("Segmentation complete!")

if st.session_state.segmented_df is not None:
    st.subheader("Retailer Segments Visualization")
    fig_clusters = px.scatter(st.session_state.segmented_df, x='total_revenue', y='overall_new_era_share', color='cluster', hover_name='store_name')
    st.plotly_chart(fig_clusters, use_container_width=True)
