import random
from supabase import create_client, Client

# --- Your Supabase Connection Info ---
url = "https://sywxhehahunevputgxdd.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"

supabase: Client = create_client(url, key)
print("Successfully connected to Supabase.")

def populate_promotion_data():
    try:
        # 1. Fetch all transaction IDs from the table
        print("Fetching transaction IDs...")
        response = supabase.table("sales_transactions").select("transaction_id").execute()
        transactions = response.data
        
        if not transactions:
            print("No transactions found.")
            return

        transaction_ids = [t['transaction_id'] for t in transactions]
        print(f"Found {len(transaction_ids)} total transactions.")

        # 2. Decide which transactions to flag as promotional (e.g., 20%)
        num_to_promote = int(len(transaction_ids) * 0.20)
        ids_to_promote = random.sample(transaction_ids, num_to_promote)
        print(f"Randomly selected {len(ids_to_promote)} transactions to mark as promotional.")

        # 3. Update the records in Supabase
        print("Updating records in the database... this may take a moment.")
        # The 'in_' filter is used to update multiple records at once
        supabase.table("sales_transactions").update({"on_promotion": True}).in_("transaction_id", ids_to_promote).execute()

        print("\n✅ Successfully populated promotion data!")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    populate_promotion_data()