# --- pages/5_🤖_SQL_Chatbot.py ---

import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import sqlalchemy

# --- Page Config ---
st.set_page_config(page_title="SQL Chatbot", layout="wide")
st.title("🤖 Chat with Your Sales Database (Text-to-SQL)")
st.markdown("---")

# --- API & DB Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    
    db_user = st.secrets["DB_USER"]
    db_password = st.secrets["DB_PASSWORD"]
    db_host = st.secrets["DB_HOST"]
    db_name = st.secrets["DB_NAME"]
    db_port = st.secrets["DB_PORT"]
    
    # Create the database connection string
    db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
except KeyError as e:
    st.error(f"A secret key is missing: {e}. Please check your .streamlit/secrets.toml file.", icon="🚨")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during configuration: {e}", icon="🚨")
    st.stop()


# --- Main App Logic ---
user_query = st.text_input("Ask a question about your sales data:", placeholder="e.g., What were the total units sold for KitKats?")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("The AI analyst is thinking and querying the database..."):
            try:
                # Setup the database connection
                db = SQLDatabase.from_uri(db_uri, include_tables=['daily_sales_summary', 'product_master', 'store_master'])
                
                # Setup the LLM
                llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
                
                # Create the SQL Database Chain
                # We add use_query_checker=True to help prevent hallucinations
                db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)
                
                # Run the query and get the result
                result = db_chain.run(user_query)
                
                st.success("Here is the answer:")
                st.markdown(result)

            except sqlalchemy.exc.OperationalError:
                st.error("Could not connect to the database. Please check your connection details in the secrets file and ensure your network allows the connection.", icon="🔥")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")