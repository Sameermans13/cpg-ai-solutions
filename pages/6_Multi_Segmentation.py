import streamlit as st
import pandas as pd
from supabase import create_client, Client
from datetime import datetime
import json
import numpy as np
import re # Import the regular expression library
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai


# --- Page Config ---
st.set_page_config(page_title="Retailer Segmentation", layout="wide")
st.title("🏬 Multi-Lens Retailer Segmentation")
st.write("Using three independent ML models to discover rich, multi-layered segments in retailer data.")
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
st.subheader("1. Store Feature Profiles")
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


# --- 2. Multi-Lens Segmentation ---
st.markdown("---")
st.subheader("2. Configure & Run Segmentation Models")
st.write("Select the number of clusters for each strategic lens and run the analysis.")

col1, col2, col3 = st.columns(3)
with col1:
    k_trend = st.number_input("Clusters for Trend Adoption:", min_value=2, max_value=8, value=3, key="k_trend")
with col2:
    k_value = st.number_input("Clusters for Business Value:", min_value=2, max_value=8, value=3, key="k_value")
with col3:
    k_mix = st.number_input("Clusters for Product Mix:", min_value=2, max_value=8, value=3, key="k_mix")

if st.button("🚀 Run All Segmentation Models"):
    with st.spinner("Running all segmentation models..."):
        segmented_df = store_features.copy()

        # Model 1: Trend Adoption
        trend_features = segmented_df[['overall_new_era_share', 'new_era_momentum']]
        scaler_trend = StandardScaler()
        scaled_features_trend = scaler_trend.fit_transform(trend_features)
        kmeans_trend = KMeans(n_clusters=k_trend, n_init='auto', random_state=42)
        segmented_df['trend_segment'] = kmeans_trend.fit_predict(scaled_features_trend)
        
        # Model 2: Business Value
        value_features = segmented_df[['total_revenue', 'revenue_growth_qoq']]
        scaler_value = StandardScaler()
        scaled_features_value = scaler_value.fit_transform(value_features)
        kmeans_value = KMeans(n_clusters=k_value, n_init='auto', random_state=42)
        # --- THE FIX IS HERE ---
        segmented_df['value_segment'] = kmeans_value.fit_predict(scaled_features_value)
        # --- END FIX ---

        # Model 3: Product Mix
        mix_features = segmented_df[['unique_products', 'overall_pct_chocolate', 'overall_pct_candy', 'overall_pct_ice_cream']]
        scaler_mix = StandardScaler()
        scaled_features_mix = scaler_mix.fit_transform(mix_features)
        kmeans_mix = KMeans(n_clusters=k_mix, n_init='auto', random_state=42)
        segmented_df['mix_segment'] = kmeans_mix.fit_predict(scaled_features_mix)
        
        st.session_state.segmented_df = segmented_df
        st.session_state.analysis_results = None # Reset AI analysis
        st.success("All three segmentations are complete!")

# --- 3. Display Results ---
if 'segmented_df' in st.session_state:
    segmented_df = st.session_state.segmented_df
    st.subheader("3. Segmentation Results")
    
    st.write("#### Final Multi-Layered Segments")
    st.dataframe(segmented_df[['store_name', 'channel_region', 'trend_segment', 'value_segment', 'mix_segment']])
    
    tab_trend, tab_value, tab_mix = st.tabs(["Trend Adoption Segments", "Value Segments", "Product Mix Segments"])
    with tab_trend:
        st.write("##### Trend Adoption Profiles")
        st.dataframe(segmented_df.groupby('trend_segment')[['overall_new_era_share', 'new_era_momentum']].mean())
        fig_trend = px.scatter(segmented_df, x='overall_new_era_share', y='new_era_momentum', color='trend_segment', hover_name='store_name')
        st.plotly_chart(fig_trend, use_container_width=True)
    with tab_value:
        st.write("##### Value Profiles")
        st.dataframe(segmented_df.groupby('value_segment')[['total_revenue', 'revenue_growth_qoq']].mean())
        fig_value = px.scatter(segmented_df, x='total_revenue', y='revenue_growth_qoq', color='value_segment', hover_name='store_name')
        st.plotly_chart(fig_value, use_container_width=True)
    with tab_mix:
        st.write("##### Product Mix Profiles")
        st.dataframe(segmented_df.groupby('mix_segment')[['unique_products', 'overall_pct_chocolate', 'overall_pct_candy', 'overall_pct_ice_cream']].mean())
        fig_mix = px.scatter(segmented_df, x='overall_pct_chocolate', y='overall_pct_ice_cream', color='mix_segment', hover_name='store_name')
        st.plotly_chart(fig_mix, use_container_width=True)

# --- 4. AI-Powered Analysis ---
st.markdown("---")
st.subheader("4. Generate Retail Stores Segmentation")

# Initialize session state variables for this section
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'final_display_df' not in st.session_state:
    st.session_state.final_display_df = None

if st.button("🤖 Generate Retail Attributes for All Segments"):
    if 'segmented_df' in st.session_state and st.session_state.segmented_df is not None:
        with st.spinner("AI is analyzing all segments... This may take a moment."):
            try:
                segmented_df = st.session_state.segmented_df.copy()
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                
                # Helper function to robustly extract JSON from AI response
                def extract_json_from_response(text_response):
                    match = re.search(r'\{.*\}', text_response, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
                    st.warning(f"AI did not return a valid JSON object. Raw response: {text_response}")
                    return None

                # --- 1. Generate & Map Trend Segment Names ---
                trend_profiles = segmented_df.groupby('trend_segment')[['overall_new_era_share', 'new_era_momentum']].mean().reset_index()
                trend_prompt = f"You are a CPG strategist. Analyze these Trend Adoption RETAILER segment profiles. For each cluster number, provide a short, professional persona name. Your output must be ONLY a valid JSON mapping object like {{\"0\": \"Persona Name 1\", \"1\": \"Persona Name 2\"}}.\n\nData:\n{trend_profiles.to_markdown(index=False)}"
                response_trend_text = llm.invoke(trend_prompt).content
                trend_map = extract_json_from_response(response_trend_text)
                if trend_map:
                    trend_map = {int(k): v for k, v in trend_map.items()}
                    segmented_df['trend_segment_name'] = segmented_df['trend_segment'].map(trend_map)

                # --- 2. Generate & Map Value Segment Names ---
                value_profiles = segmented_df.groupby('value_segment')[['total_revenue', 'revenue_growth_qoq']].mean().reset_index()
                value_prompt = f"You are a CPG strategist. Analyze these Business Value RETAILER segment profiles. For each cluster number, provide a short, professional persona name. Your output must be ONLY a valid JSON mapping object like {{\"0\": \"Persona Name 1\"}}.\n\nData:\n{value_profiles.to_markdown(index=False)}"
                response_value_text = llm.invoke(value_prompt).content
                value_map = extract_json_from_response(response_value_text)
                if value_map:
                    value_map = {int(k): v for k, v in value_map.items()}
                    segmented_df['value_segment_name'] = segmented_df['value_segment'].map(value_map)

                # --- 3. Generate & Map Mix Segment Names ---
                mix_profiles = segmented_df.groupby('mix_segment')[['unique_products', 'overall_pct_chocolate', 'overall_pct_candy', 'overall_pct_ice_cream']].mean().reset_index()
                mix_prompt = f"You are a CPG strategist. Analyze these RETAILER product mix segments. For each cluster number, provide a short, professional persona name. Your output must be ONLY a valid JSON mapping object like {{\"0\": \"Persona Name 1\"}}.\n\nData:\n{mix_profiles.to_markdown(index=False)}"
                response_mix_text = llm.invoke(mix_prompt).content
                mix_map = extract_json_from_response(response_mix_text)
                if mix_map:
                    mix_map = {int(k): v for k, v in mix_map.items()}
                    segmented_df['mix_segment_name'] = segmented_df['mix_segment'].map(mix_map)
                
                st.session_state.final_display_df = segmented_df
                st.success("AI Analysis and Mapping Complete!")
            
            except Exception as e:
                st.error(f"An error occurred during AI analysis: {e}")
    else:
        st.warning("Please run the segmentation first to generate the data needed for analysis.")

# --- Display Final Results ---
if 'final_display_df' in st.session_state and st.session_state.final_display_df is not None:
    final_df = st.session_state.final_display_df
    st.subheader("Final Multi-Layered Segments for Each Retailer")
    st.dataframe(final_df[['store_name', 'channel_region', 'trend_segment_name', 'value_segment_name', 'mix_segment_name']])
    
    # --- 5. Save to Database ---
    st.markdown("---")
    st.subheader("5. Save Final Segments to Database")
    if st.button("💾 Save Final Segments to Supabase"):
        with st.spinner("Updating database..."):
            try:
                service_key = st.secrets["SUPABASE_SERVICE_KEY"]
                url = st.secrets["SUPABASE_URL"]
                supabase_admin = create_client(url, service_key)
                
                # Update the master table with the new text-based segment labels
                for _, row in final_df.iterrows():
                    supabase_admin.table('store_master').update({
                        'trend_segment': row['trend_segment_name'],
                        'value_segment': row['value_segment_name'],
                        'mix_segment': row['mix_segment_name']
                    }).eq('store_id', row['store_id']).execute()
                
                st.success("Successfully saved all three segment labels for each store to Supabase!")

            except Exception as e:
                st.error(f"An error occurred while saving to the database: {e}")


# --- 6. AI-Powered Segment Analysis---
st.markdown("---")
st.subheader("6. AI-Powered Segment Analysis")
st.write("Use a Generative AI to translate the numerical cluster profiles into strategic business personas for each of the three segmentation models.")

# This entire section will only appear if the segmentation has been run
if 'segmented_df' in st.session_state and st.session_state.segmented_df is not None:
    
    if st.button("🤖 Generate AI Personas for All Segments"):
        with st.spinner("AI is analyzing all segments... This may take a moment."):
            try:
                segmented_df = st.session_state.segmented_df
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.4)
                
                # --- 1. Analyze Trend Segments ---
                st.write("### Trend Adoption Personas")
                trend_profiles = segmented_df.groupby('trend_segment')[['overall_new_era_share', 'new_era_momentum']].mean().to_markdown()
                trend_prompt = f"""
                You are a CPG business consultant. Analyze the following RETAILER segment profiles based on their adoption of 'New Era' products. 
                Based ONLY on the provided data, create a persona for each cluster. 
                Do not invent consumer demographics. Focus on the business characteristics.

                Data:
                {trend_profiles}
                """
                response_trend = llm.invoke(trend_prompt).content
                st.markdown(response_trend)
                
                # --- 2. Analyze Value Segments ---
                st.write("### Business Value Personas")
                value_profiles = segmented_df.groupby('value_segment')[['total_revenue', 'revenue_growth_qoq']].mean().to_markdown()
                value_prompt = f"""
                You are a CPG business consultant. Analyze the following RETAILER segment profiles based on their business value. 
                Based ONLY on the data, create a persona for each cluster. 
                Do not invent details. Describe their business value and growth trajectory.

                Data:
                {value_profiles}
                """
                response_value = llm.invoke(value_prompt).content
                st.markdown(response_value)
                
                # --- 3. Analyze Mix Segments ---
                st.write("### Product Mix Personas")
                mix_profiles = segmented_df.groupby('mix_segment')[['unique_products', 'overall_pct_chocolate', 'overall_pct_candy', 'overall_pct_ice_cream']].mean().to_markdown()
                mix_prompt = f"""
                You are a CPG business consultant. Analyze these RETAILER product mix segments. 
                Based ONLY on the data, create a persona for each cluster. 
                Do not invent details. Describe their product focus and assortment strategy.

                Data:
                {mix_profiles}
                """
                response_mix = llm.invoke(mix_prompt).content
                st.markdown(response_mix)

                st.success("AI Analysis Complete!")

            except Exception as e:
                st.error(f"An error occurred during AI analysis: {e}")
else:
    st.warning("Please run the main segmentation models first to enable AI analysis.")
