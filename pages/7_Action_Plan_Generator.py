# --- pages/6_🎯_Action_Plan_Generator.py ---

import streamlit as st
import pandas as pd
from supabase import create_client, Client
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Action Plan Generator", layout="wide")
st.title("🎯 Sales Activity Recommendation Engine")
st.write("This tool uses the retailer segmentation models to recommend the top 3 most impactful activities for a salesperson's store visit.")
st.markdown("---")

# --- Supabase Connection ---
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
except Exception as e:
    st.error(f"Configuration error: {e}. Please check your secrets.", icon="🚨")
    st.stop()

# --- Data Loading ---
@st.cache_data
def load_master_data():
    """Loads all master data tables from Supabase."""
    stores = supabase.table("store_master").select("*").execute().data
    activities = supabase.table("activity_master").select("*").execute().data
    return pd.DataFrame(stores), pd.DataFrame(activities)

df_stores, df_activities = load_master_data()


# --- Main Application ---

# 1. User selects a store
st.subheader("Step 1: Select a Retailer")
store_name_list = df_stores['store_name'].unique()
selected_store_name = st.selectbox(
    "Select a store to generate an action plan for:",
    options=store_name_list
)

if selected_store_name:
    # 2. Fetch and display the selected store's profile
    store_profile = df_stores[df_stores['store_name'] == selected_store_name].iloc[0]
    
    st.subheader(f"Profile for: {selected_store_name}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Trend Segment", store_profile['trend_segment'])
    with col2:
        st.metric("Value Segment", store_profile['value_segment'])
    with col3:
        st.metric("Mix Segment", store_profile['mix_segment'])
        
    st.markdown("---")

    # 3. Filter activities for the current month
    current_month = datetime.today().month
    
    st.subheader(f"Step 2: Potential Activities for the Current Month ({datetime.today().strftime('%B')})")
    
    monthly_activities = df_activities[df_activities['applicable_month'] == current_month].copy()
    
    if monthly_activities.empty:
        st.warning("No specific activities found for the current month.")
    else:
        st.dataframe(monthly_activities[['activity_description', 'brand', 'product_era']])

# --- ADD THIS FINAL SECTION TO YOUR SCRIPT ---

# This code will only execute if a store has been selected
if selected_store_name:
    st.markdown("---")
    st.subheader("Step 3: Generate Top 3 Recommendations")

    # Define the scoring function (the "brain")
    def score_activities(activities, profile):
        scores = []
        for _, activity in activities.iterrows():
            score = 0
            # Rule 1: Value Segment - Growth Focus
            if "Growth" in profile['value_segment'] and activity['product_era'] == 'New Era':
                score += 15
            
            # Rule 2: Value Segment - Stability Focus
            if "Stable" in profile['value_segment'] and activity['product_era'] == 'Traditional':
                score += 10

            # Rule 3: Trend Segment - Early Adopter Focus
            if "Adopter" in profile['trend_segment'] and activity['product_era'] == 'New Era':
                score += 15
            
            # Rule 4: Trend Segment - Traditional Focus
            if "Traditional" in profile['trend_segment'] and activity['product_era'] == 'Traditional':
                score += 10
            
            # Rule 5: Mix Segment - Category Focus
            if "Chocolate" in profile['mix_segment'] and activity['brand'] in ["Hershey's", "Reese's", "KitKat"]:
                score += 12
            if "Ice Cream" in profile['mix_segment'] and activity['product_category'] == 'Ice Cream':
                score += 12
            if "Candy" in profile['mix_segment'] and activity['brand'] in ["Twizzlers", "Jolly Rancher"]:
                score += 12

            scores.append(score)
        
        activities['score'] = scores
        return activities

    if st.button("Generate Top 3 Recommendations"):
        with st.spinner("Analyzing profile and scoring activities..."):
            # Run the scoring engine
            scored_activities = score_activities(monthly_activities, store_profile)
            
            # Get the top 3 activities
            top_3_activities = scored_activities.sort_values(by='score', ascending=False).head(3)

            st.write("#### Recommended Action Plan:")
            st.dataframe(top_3_activities[['activity_description', 'brand', 'product_era', 'score']])
