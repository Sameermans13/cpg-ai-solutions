# --- pages/2_📈_Forecasting.py ---
import streamlit as st
import pandas as pd
from supabase import create_client, Client
from datetime import datetime
import holidays
from prophet import Prophet
import google.generativeai as genai

# --- SHARED CODE ---
st.set_page_config(page_title="Demand Forecasting", layout="wide")

# (Your full, shared load_data function and Supabase connection should be here)
# ...
@st.cache_data
def load_data():
    # ... (Your complete load_data function code) ...
    with st.spinner('Loading summarized sales data from Supabase...'):
        products = supabase.table("product_master").select("*").execute().data
        df_products = pd.DataFrame(products)
        daily_summary = supabase.table("daily_sales_summary").select("*").execute().data
        df_sales = pd.DataFrame(daily_summary)
    df_sales.rename(columns={'sale_date': 'created_at','total_units_sold': 'units_sold','average_sale_price': 'average_sale_price','was_on_promotion': 'on_promotion'}, inplace=True)
    df_sales["created_at"] = pd.to_datetime(df_sales["created_at"], utc=True)
    today = pd.to_datetime('today').tz_localize('UTC')
    df_sales = df_sales[df_sales['created_at'] <= today].copy()
    df_sales['on_promotion'] = df_sales['on_promotion'].fillna(False).astype(bool)
    df_sales['day_of_week'] = df_sales['created_at'].dt.day_name()
    df_sales['month'] = df_sales['created_at'].dt.month
    df_sales['week_of_year'] = df_sales['created_at'].dt.isocalendar().week
    us_holidays = holidays.US(years=df_sales['created_at'].dt.year.unique())
    df_sales['is_holiday'] = df_sales['created_at'].dt.date.isin(us_holidays)
    return df_products, df_sales

# Supabase Connection
url = "https://sywxhehahunevputgxdd.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"
supabase: Client = create_client(url, key)

df_products, df_sales = load_data()

# --- THE FIX: A Clean Merge ---
st.title("📈 Demand Forecasting")
st.write("Select a product to train a model and predict future sales.")

# 1. Prepare df_products by dropping its unnecessary 'created_at' column
if 'created_at' in df_products.columns:
    df_products_to_merge = df_products.drop(columns=['created_at'])
else:
    df_products_to_merge = df_products

# 2. Merge the full sales data with the cleaned product details
df_merged = df_sales.merge(df_products_to_merge, on="product_id", how="left")


# ----------------------------------
# Demand Forecasting Section
# ----------------------------------
st.subheader("Future Demand Forecasting")
st.markdown("---") 

# --- API KEY CONFIGURATION ---
# This line reads the secret key you stored.
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

product_list = df_products['product_name'].unique()
selected_product = st.selectbox(
    "Select a Product to Forecast",
    options=product_list
)

if selected_product:
    df_product_forecast = df_merged[df_merged['product_name'] == selected_product].copy()
    df_prophet = df_product_forecast.rename(columns={"created_at": "ds", "units_sold": "y"})
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    with st.spinner(f"Training model and generating forecast for {selected_product}..."):
        holidays_df = df_prophet[df_prophet['is_holiday'] == True][['ds']].copy()
        holidays_df['holiday'] = 'US Holiday'
        model = Prophet(holidays=holidays_df)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
    
    st.success("Forecast complete!")

    # --- ACTIONABLE INSIGHTS & BUSINESS SUMMARY ---
    st.subheader("📈 Actionable Insights & Business Summary")

    future_forecast = forecast[forecast['ds'] > df_prophet['ds'].max()]
    forecast_30_days = int(future_forecast.head(30)['yhat'].sum())
    forecast_60_days = int(future_forecast.head(60)['yhat'].sum())
    forecast_90_days = int(future_forecast['yhat'].sum())
    trend_start = future_forecast.head(1)['trend'].iloc[0]
    trend_end = future_forecast.tail(1)['trend'].iloc[0]
    trend_growth_pct = round(((trend_end - trend_start) / trend_start) * 100, 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Next 30-Day Forecast (Units)", f"{forecast_30_days:,}")
    col2.metric("Next 60-Day Forecast (Units)", f"{forecast_60_days:,}")
    col3.metric("Next 90-Day Forecast (Units)", f"{forecast_90_days:,}")
    
    st.metric("Underlying 90-Day Growth Trend", f"{trend_growth_pct}%")

    # --- NEW: GENERATIVE AI SUMMARY ---
    if st.button("🤖 Generate AI Business Summary"):
        with st.spinner("AI is drafting the summary..."):
            # 1. Create the Prompt
            prompt = f"""
            You are a senior CPG business analyst reporting to an executive.
            Analyze the following sales forecast data for the product '{selected_product}' and write a concise, professional business summary in markdown format.

            Data:
            - Next 30-Day Forecasted Sales: {forecast_30_days:,} units
            - Next 60-Day Forecasted Sales: {forecast_60_days:,} units
            - Next 90-Day Forecasted Sales: {forecast_90_days:,} units
            - Underlying 90-Day Growth Trend: {trend_growth_pct}%

            Your summary should:
            - Start with a clear headline.
            - State the key findings in bullet points.
            - Provide a brief interpretation of the trend (is the growth healthy, stagnant, or a concern?).
            - Conclude with a clear recommendation for the next step (e.g., inventory planning, marketing strategy).
            """
            
            # 2. Call the Generative Model
            generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = generative_model.generate_content(prompt)
            
            # 3. Display the result
            st.markdown(response.text)

    # --- VISUALIZATIONS ---
    st.subheader("Forecast Visualization")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)