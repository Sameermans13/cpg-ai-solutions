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
st.write("Using unsupervised ML to discover hidden segments based on performance, product mix, and trends.")
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
    df_sales['day_of_week'] = df_sales['created_at'].dt.day_name()
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
st.write("Creating a rich profile for each store, including overall performance, product mix, geography, and dynamic trends.")

df_stores['channel_region'] = df_stores['store_channel'] + '_' + df_stores['store_region']
df_sales['revenue'] = df_sales['units_sold'] * df_sales['average_sale_price']
df_sales_merged = df_sales.merge(df_products[['product_id', 'product_category', 'product_era']], on='product_id', how='left')

# Overall Features
category_sales_overall = df_sales_merged.pivot_table(index='store_id', columns='product_category', values='units_sold', aggfunc='sum', fill_value=0)
category_sales_pct_overall = category_sales_overall.div(category_sales_overall.sum(axis=1), axis=0).fillna(0)
category_sales_pct_overall.columns = [f'overall_pct_{c.lower().replace(" ", "_")}' for c in category_sales_pct_overall.columns]
era_sales_overall = df_sales_merged.pivot_table(index='store_id', columns='product_era', values='units_sold', aggfunc='sum', fill_value=0)
if 'New Era' not in era_sales_overall.columns: era_sales_overall['New Era'] = 0
era_sales_pct_overall = (era_sales_overall['New Era'] / (era_sales_overall.sum(axis=1) + 0.001)).to_frame('overall_new_era_share').fillna(0)
store_features = df_sales_merged.groupby('store_id').agg(total_revenue=('revenue', 'sum'), total_units=('units_sold', 'sum'), unique_products=('product_id', 'nunique')).reset_index()
store_features = store_features.merge(category_sales_pct_overall, on='store_id', how='left')
store_features = store_features.merge(era_sales_pct_overall, on='store_id', how='left')

# Trend Features
last_date = df_sales_merged['created_at'].max()
recent_start_date = last_date - pd.Timedelta(days=90)
previous_start_date = last_date - pd.Timedelta(days=180)
df_recent = df_sales_merged[df_sales_merged['created_at'] >= recent_start_date]
df_previous = df_sales_merged[(df_sales_merged['created_at'] >= previous_start_date) & (df_sales_merged['created_at'] < recent_start_date)]
revenue_recent = df_recent.groupby('store_id')['revenue'].sum().reset_index().rename(columns={'revenue': 'revenue_last_90d'})
revenue_previous = df_previous.groupby('store_id')['revenue'].sum().reset_index().rename(columns={'revenue': 'revenue_prev_90d'})
store_features = store_features.merge(revenue_recent, on='store_id', how='left')
store_features = store_features.merge(revenue_previous, on='store_id', how='left')
store_features.fillna(0, inplace=True)
store_features['revenue_growth_qoq'] = (store_features['revenue_last_90d'] - store_features['revenue_prev_90d']) / (store_features['revenue_prev_90d'] + 1)
era_sales_recent = df_recent.pivot_table(index='store_id', columns='product_era', values='units_sold', aggfunc='sum', fill_value=0)
if 'New Era' not in era_sales_recent.columns: era_sales_recent['New Era'] = 0
recent_new_era_share = (era_sales_recent['New Era'] / (era_sales_recent.sum(axis=1) + 0.001)).to_frame('recent_new_era_share').fillna(0)
store_features = store_features.merge(recent_new_era_share, on='store_id', how='left').fillna(0)
store_features['new_era_momentum'] = store_features['recent_new_era_share'] - store_features['overall_new_era_share']

# Final Merge
store_features = store_features.merge(df_stores[['store_id', 'store_name', 'channel_region']], on='store_id', how='left')
st.dataframe(store_features)


# --- 2. Feature Scaling & Elbow Method ---
st.markdown("---")
st.subheader("2. Finding the Optimal Number of Clusters (K)")
features_for_clustering = store_features.select_dtypes(include=np.number).drop(columns=['store_id'])
if not features_for_clustering.empty:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)
    inertia = []
    k_range = range(1, 11)
    with st.spinner("Calculating optimal clusters..."):
        for k in k_range:
            if k <= len(store_features):
                kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
                kmeans.fit(scaled_features)
                inertia.append(kmeans.inertia_)
    if inertia:
        fig = go.Figure(data=go.Scatter(x=list(k_range), y=inertia, mode='lines+markers'))
        fig.update_layout(title='Elbow Method for Optimal K', xaxis_title='Number of Clusters (K)', yaxis_title='Inertia')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Look for the 'elbow' in the chart to determine the best 'K' for the next step.", icon="🎯")

# --- 3. Final Segmentation and Analysis ---
st.markdown("---")
st.subheader("3. Final Segmentation and Analysis")

if 'cluster_profiles' not in st.session_state: st.session_state.cluster_profiles = None
if 'segmented_data' not in st.session_state: st.session_state.segmented_data = None
if 'final_crm_data' not in st.session_state: st.session_state.final_crm_data = None

if 'scaled_features' in locals():
    optimal_k = st.number_input("Based on the elbow chart, select the optimal number of clusters (K):", min_value=2, max_value=10, value=4, step=1)
    if st.button("Run Final Segmentation"):
        with st.spinner("Training final model and creating segments..."):
            kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
            kmeans.fit(scaled_features)
            results_df = store_features.copy()
            results_df['cluster'] = kmeans.labels_
            cluster_profiles = results_df.groupby('cluster').mean(numeric_only=True)
            st.session_state.cluster_profiles = cluster_profiles
            st.session_state.segmented_data = results_df
            st.session_state.final_crm_data = None 
            st.success("Segmentation complete!")

if st.session_state.segmented_data is not None:
    st.subheader("Retailer Segments Visualization")
    fig_clusters = px.scatter(st.session_state.segmented_data, x='total_revenue', y='overall_new_era_share', color='cluster', hover_name='store_name', title=f'Retailer Segments (K={optimal_k})')
    st.plotly_chart(fig_clusters, use_container_width=True)
    st.subheader("Cluster Profiles")
    st.dataframe(st.session_state.cluster_profiles)

    # --- 4. AI-Powered Analysis and Operationalization ---
    st.markdown("---")
    st.subheader("4. AI-Powered Segment Analysis & CRM Preparation")
    if st.button("🤖 Generate Full Segment Analysis"):
        with st.spinner("AI is analyzing segments, drafting personas, and preparing CRM data..."):
            try:
                profiles_with_cluster_id = st.session_state.cluster_profiles.reset_index()
                profiles_markdown = profiles_with_cluster_id.to_markdown(index=False)
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                unified_prompt = f"""
                You are a senior CPG marketing strategist. Your task is to analyze the following table of retailer segment profiles and generate a single, valid JSON array. Each object in the array must represent one cluster.
                **Cluster Profiles Data:**
                {profiles_markdown}
                **Instructions for each cluster object:**
                - Create these exact five keys: "cluster", "segment_name", "persona_description", "primary_focus_era", "focus_intensity".
                - "segment_name": A short, professional persona name (e.g., "Traditional Strongholds").
                - "persona_description": A markdown string with 2-3 bullet points describing the segment's key characteristics.
                - "primary_focus_era": CRITICAL RULE: Analyze 'overall_new_era_share' and 'new_era_momentum'. If 'overall_new_era_share' is below 0.3 AND 'new_era_momentum' is negative or near zero (e.g., less than 0.02), you MUST classify the focus as 'Traditional'. Otherwise, classify it as 'New Era'.
                - "focus_intensity": Assign 'High', 'Medium', or 'Low' based on growth and momentum.
                - Your final output must be ONLY the JSON array and nothing else.
                """
                response = llm.invoke(unified_prompt)
                json_string = response.content.strip().replace("```json", "").replace("```", "")
                analysis_results = json.loads(json_string)
                df_analysis = pd.DataFrame(analysis_results)
                final_crm_data = st.session_state.segmented_data.merge(df_analysis, on='cluster', how='left')
                st.session_state.final_crm_data = final_crm_data
                st.success("Full analysis complete!")
            except Exception as e:
                st.error(f"An error occurred during AI analysis: {e}")

if st.session_state.final_crm_data is not None:
    st.subheader("AI-Generated Segment Analysis")
    for _, row in st.session_state.final_crm_data.drop_duplicates(subset=['cluster']).iterrows():
        st.write(f"**Segment: {row['segment_name']} (Cluster {row['cluster']})**")
        st.markdown(row['persona_description'])
        st.write(f"**Primary Focus**: {row['primary_focus_era']} ({row['focus_intensity']} Intensity)")
        st.markdown("---")

    st.subheader("Final CRM-Ready Data")
    st.dataframe(st.session_state.final_crm_data[['store_name', 'segment_name', 'primary_focus_era', 'focus_intensity']])
    
    # --- 5. Save to Supabase ---
    st.subheader("Save Results to CRM")
    if st.button("💾 Save Results to CRM"):
        with st.spinner("Updating database..."):
            try:
                service_key = st.secrets["SUPABASE_SERVICE_KEY"]
                url = st.secrets["SUPABASE_URL"]
                supabase_admin = create_client(url, service_key)
                data_to_save = st.session_state.final_crm_data
                history_data = data_to_save[['store_id', 'segment_name', 'primary_focus_era', 'focus_intensity']].copy()
                history_data['run_date'] = datetime.today().date().isoformat()
                supabase_admin.table('segmentation_history').insert(history_data.to_dict(orient='records')).execute()
                for _, row in data_to_save.iterrows():
                    supabase_admin.table('store_master').update({'segment_name': row['segment_name'], 'primary_focus_era': row['primary_focus_era'], 'focus_intensity': row['focus_intensity']}).eq('store_id', row['store_id']).execute()
                st.success("Successfully saved results to CRM!")
            except Exception as e:
                st.error(f"An error occurred while saving: {e}")
