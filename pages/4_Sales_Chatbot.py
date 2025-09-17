# --- pages/5_🤖_SQL_Chatbot.py ---

import streamlit as st
from supabase import create_client, Client
import google.generativeai as genai
# Add 'date' to your datetime import at the top of the file
from datetime import datetime, date


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is correct, otherwise prompt for the password.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Enter Password to Access", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("😕 Password incorrect")
    return False


if check_password():

    # --- Page Config ---
    st.set_page_config(page_title="SQL Chatbot", layout="wide")
    st.title("🤖 Chat with Your Sales Data")
    st.markdown("---")

    # --- API & DB Configuration ---
    try:
        # Use st.secrets for Supabase connection as well
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        supabase: Client = create_client(url, key)
        
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception as e:
        st.error(f"A configuration error occurred: {e}. Please check your secrets.", icon="🚨")
        st.stop()

    # --- Main App Logic ---
    st.write("Ask a question in plain English, and the AI will translate it to SQL, query the database, and give you the answer.")

    user_query = st.text_input("Your question:", placeholder="e.g., Which store has the highest total revenue?")

    if st.button("Get Answer"):
        if user_query:
            with st.spinner("The AI analyst is thinking..."):
                try:
                    # 1. Generate the SQL query using an improved, date-aware prompt
                    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
                    
                    # Get today's date to pass into the prompt
                    current_date = date.today().strftime('%Y-%m-%d')

                    # --- NEW: CONTEXT-AWARE PROMPT ---
                    schema_prompt = f"""
                    You are a world-class SQL writer. Given a user question, you will write a single, valid PostgreSQL query to answer it.
                    You must only write the SQL query and nothing else.

                    IMPORTANT RULES:
                    - The current date is {current_date}. All queries MUST include a WHERE clause to filter for dates on or before this date. Use `sale_date <= '{current_date}'`.
                    - Ensure proper SQL syntax with spaces between all keywords and table/column names.
                    - DO NOT include a trailing semicolon at the end of the query.

                    My database has the following tables and columns:
                    1. daily_sales_summary (sale_date, product_id, store_id, total_units_sold, average_sale_price, was_on_promotion)
                    2. product_master (product_id, product_name, product_category, retail_price)
                    3. store_master (store_id, store_name, store_channel, store_region)

                    --- EXAMPLES ---
                    -- Example 1: Find the total sales for a specific product.
                    User Question: "How many units of KitKat 4-finger were sold?"
                    SQL Query:
                    SELECT SUM(t1.total_units_sold) FROM daily_sales_summary AS t1 JOIN product_master AS t2 ON t1.product_id = t2.product_id WHERE t2.product_name = 'KitKat 4-finger' AND t1.sale_date <= '{current_date}'

                    -- Example 2: Find the total revenue for a brand.
                    User Question: "What was the total revenue for all Reese's products?"
                    SQL Query:
                    SELECT SUM(t1.total_units_sold * t1.average_sale_price) FROM daily_sales_summary AS t1 JOIN product_master AS t2 ON t1.product_id = t2.product_id WHERE t2.product_name LIKE 'Reese''s%' AND t1.sale_date <= '{current_date}'
                    
                    --- END EXAMPLES ---

                    User Question: "{user_query}"
                    SQL Query:
                    """
                    
                    response = llm.generate_content(schema_prompt)
                    sql_query = response.text.strip()

                    if sql_query.endswith(';'):
                        sql_query = sql_query[:-1]

                    st.write("🔍 Generated SQL Query:")
                    st.code(sql_query, language="sql")

                    # 2. Execute the query
                    with st.spinner("Executing query against the database..."):
                        rpc_response = supabase.rpc('execute_sql', {'query': sql_query}).execute()
                        query_result = rpc_response.data

                    # 3. Formulate final answer
                    with st.spinner("Formulating final answer..."):
                        final_prompt = f"""
                        You are a helpful CPG business analyst. Given a user's question and the result from a database query, formulate a clear, human-readable answer.

                        User Question: "{user_query}"
                        Query Result: {query_result}
                        Final Answer:
                        """
                        final_response = llm.generate_content(final_prompt)
                        
                        st.success("Here is the answer:")
                        st.markdown(final_response.text)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question.")