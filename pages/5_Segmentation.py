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
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI


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


# --- 1. Gold Standard Data Preparation with All Features ---
st.subheader("1. Store Performance Metrics")
st.write("Creating a rich profile for each store, including overall performance, product mix, geographic attributes, and dynamic trends.")

# Ensure we have the latest product master with the 'product_era' column
products = supabase.table("product_master").select("*").execute().data
df_products = pd.DataFrame(products)

# --- Create Geographic Interaction Feature ---
df_stores['channel_region'] = df_stores['store_channel'] + '_' + df_stores['store_region']

# --- Merge all master data with sales data ---
df_sales['revenue'] = df_sales['units_sold'] * df_sales['average_sale_price']
df_sales_merged = df_sales.merge(df_products[['product_id', 'product_category', 'product_era']], on='product_id', how='left')

# --- Calculate Overall Historical Features ---
# 1a. Overall Product Mix (by category)
category_sales_overall = df_sales_merged.pivot_table(index='store_id', columns='product_category', values='units_sold', aggfunc='sum', fill_value=0)
category_sales_pct_overall = category_sales_overall.div(category_sales_overall.sum(axis=1), axis=0).fillna(0)
category_sales_pct_overall.columns = [f'overall_pct_{col.lower().replace(" ", "_")}' for col in category_sales_pct_overall.columns]

# 1b. Overall Era Mix (New vs. Traditional)
era_sales_overall = df_sales_merged.pivot_table(index='store_id', columns='product_era', values='units_sold', aggfunc='sum', fill_value=0)
era_sales_pct_overall = (era_sales_overall['New Era'] / (era_sales_overall.sum(axis=1) + 0.001)).to_frame('overall_new_era_share').fillna(0)

# 1c. Overall Core Metrics
store_features = df_sales_merged.groupby('store_id').agg(
    total_revenue=('revenue', 'sum'),
    total_units=('units_sold', 'sum'),
    unique_products=('product_id', 'nunique'),
).reset_index()

# --- THE FIXES ARE IN THE MERGE COMMANDS BELOW ---
store_features = store_features.merge(category_sales_pct_overall, on='store_id', how='left')
store_features = store_features.merge(era_sales_pct_overall, on='store_id', how='left')


# --- Calculate Recent Performance & Trend Features ---
last_date = df_sales_merged['created_at'].max()
recent_start_date = last_date - pd.Timedelta(days=90)
previous_start_date = last_date - pd.Timedelta(days=180)

df_recent = df_sales_merged[df_sales_merged['created_at'] >= recent_start_date]
df_previous = df_sales_merged[(df_sales_merged['created_at'] >= previous_start_date) & (df_sales_merged['created_at'] < recent_start_date)]

# 2a. Revenue Growth Trend
revenue_recent = df_recent.groupby('store_id')['revenue'].sum().reset_index().rename(columns={'revenue': 'revenue_last_90d'})
revenue_previous = df_previous.groupby('store_id')['revenue'].sum().reset_index().rename(columns={'revenue': 'revenue_prev_90d'})
store_features = store_features.merge(revenue_recent, on='store_id', how='left')
store_features = store_features.merge(revenue_previous, on='store_id', how='left')
store_features.fillna(0, inplace=True)
store_features['revenue_growth_qoq'] = (store_features['revenue_last_90d'] - store_features['revenue_prev_90d']) / (store_features['revenue_prev_90d'] + 1)

# 2b. New Era Momentum Trend
era_sales_recent = df_recent.pivot_table(index='store_id', columns='product_era', values='units_sold', aggfunc='sum', fill_value=0)
if 'New Era' not in era_sales_recent.columns: era_sales_recent['New Era'] = 0
recent_new_era_share = (era_sales_recent['New Era'] / (era_sales_recent.sum(axis=1) + 0.001)).to_frame('recent_new_era_share').fillna(0)
store_features = store_features.merge(recent_new_era_share, on='store_id', how='left').fillna(0)
store_features['new_era_momentum'] = store_features['recent_new_era_share'] - store_features['overall_new_era_share']


# --- Final Merge with Store Details ---
store_features = store_features.merge(df_stores[['store_id', 'store_name', 'store_channel', 'channel_region']], on='store_id', how='left')

st.dataframe(store_features)


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




# --- 3. Train Final Model & Analyze Segments ---
st.markdown("---")
st.subheader("3. Final Segmentation and Analysis")

# Initialize cluster_profiles in session state if it doesn't exist
if 'cluster_profiles' not in st.session_state:
    st.session_state.cluster_profiles = None

optimal_k = st.number_input(
    "Select the optimal number of clusters (K) based on the elbow chart:",
    min_value=2,
    max_value=10,
    value=4, # A common default, adjust based on your chart
    step=1
)

if st.button("Run Segmentation"):
    # Check if scaled_features exists and is not empty
    if 'scaled_features' in locals() and scaled_features.shape[0] > 0:
        with st.spinner("Training final model and creating segments..."):
            kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
            kmeans.fit(scaled_features)
            
            # Create a fresh copy to avoid modifying the original dataframe
            results_df = store_features.copy()
            results_df['cluster'] = kmeans.labels_
            
            # Calculate the cluster profiles
            cluster_profiles = results_df.groupby('cluster').mean(numeric_only=True)
            
            # --- SAVE TO SESSION STATE ---
            st.session_state.cluster_profiles = cluster_profiles
            st.session_state.segmented_data = results_df
            
            st.success("Segmentation complete!")
    else:
        st.warning("Feature data is not available. Please ensure the steps above have run correctly.")

# --- Display results ONLY if they exist in session state ---
if st.session_state.cluster_profiles is not None:
    st.subheader("Retailer Segments Visualization")
    fig_clusters = px.scatter(
        st.session_state.segmented_data,
        x='total_revenue',
        y='total_units',
        color='cluster',
        hover_name='store_name',
        hover_data=['store_channel', 'unique_products'],
        title=f'Retailer Segments (K={optimal_k})',
        color_continuous_scale=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig_clusters, use_container_width=True)
    
    st.subheader("Cluster Profiles")
    st.dataframe(st.session_state.cluster_profiles)

    # --- 4. AI-Powered Segment Analysis ---
    st.subheader("4. AI-Powered Segment Analysis")
    if st.button("🤖 Generate AI Analysis"):
        with st.spinner("AI is analyzing the segments and drafting personas..."):
            try:
                # Read the data FROM session state
                profiles_markdown = st.session_state.cluster_profiles.to_markdown()
                
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)

                analysis_prompt = f"""
                You are a senior CPG marketing strategist...
                **Cluster Profiles Data:**
                {profiles_markdown}
                For each cluster... provide a Persona Name, Key Characteristics, and a Strategic Action.
                """
                
                response = llm.invoke(analysis_prompt)
                st.markdown(response.content)

            except Exception as e:
                st.error(f"An error occurred: {e}")



# --- 5. Generate CRM-Ready Attributes ---
st.markdown("---")
st.subheader("5. Generate CRM-Ready Attributes")
st.write("This tool uses the AI to distill the complex segment profiles into simple, actionable tags that can be uploaded to a CRM to guide sales activities.")

# Check if the cluster profiles have been generated and stored in session state
if 'cluster_profiles' in st.session_state and st.session_state.cluster_profiles is not None:
    if st.button("🏷️ Generate CRM Attributes"):
        with st.spinner("AI is distilling the analysis into CRM tags..."):
            try:
                # Prepare the data for the prompt
                profiles_markdown = st.session_state.cluster_profiles.to_markdown()
                
                # Configure the LLM
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)

                # Engineer the new "distiller" prompt
                crm_prompt = f"""
                You are a data analyst preparing a file for upload into a CRM system. Your task is to analyze the following table of retailer segment profiles and distill it into a structured JSON output.

                **Cluster Profiles Data:**
                {profiles_markdown}

                Based on the data, particularly the 'overall_new_era_share' and 'new_era_momentum' columns, your task is to generate a JSON object for each cluster with the following three attributes:
                1. "segment_name": A short, descriptive name for the segment (e.g., "New Era Growth Market").
                2. "primary_focus_era": Identify if the primary strategic focus for this segment should be on 'New Era' or 'Traditional' products. The value must be ONLY one of these two strings.
                3. "focus_intensity": Based on the momentum and growth, assign an intensity level. The value must be ONLY one of 'High', 'Medium', or 'Low'.

                Your final output must be a single, valid JSON array containing the objects for each cluster, and absolutely nothing else. Do not include any explanatory text or markdown formatting.
                
                Example of a single JSON object:
                {{
                  "segment_name": "Example Segment",
                  "primary_focus_era": "New Era",
                  "focus_intensity": "High"
                }}
                """
                
                # Invoke the AI
                response = llm.invoke(crm_prompt)
                
                # --- NEW: Parse the AI's JSON output ---
                import json
                
                # The response content will be a string that looks like a JSON array.
                # We need to clean it and parse it.
                json_string = response.content.strip().replace("```json", "").replace("```", "")
                crm_attributes = json.loads(json_string)
                
                # Convert the list of dictionaries to a Pandas DataFrame for display
                df_crm = pd.DataFrame(crm_attributes)
                
                st.success("CRM attributes generated successfully!")
                st.dataframe(df_crm)

                st.download_button(
                   label="Download CRM Data as CSV",
                   data=df_crm.to_csv(index=False).encode('utf-8'),
                   file_name='crm_segment_attributes.csv',
                   mime='text/csv',
                )

            except Exception as e:
                st.error(f"An error occurred while generating CRM attributes: {e}")
else:
    st.warning("Please run the segmentation first to generate cluster profiles.")
