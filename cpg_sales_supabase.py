from supabase import create_client, Client

url = "https://sywxhehahunevputgxdd.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"

supabase = create_client(url, key)

# Fetch all rows from product_master
read_response = supabase.table("product_master").select("*", count="exact").execute()

print("All products:", read_response.data)
print("Number of rows:", read_response.count)

# Insert a new product
#insert_response = supabase.table("product_master").insert(
#    {"product_id": 201, "product_name": "KitKat 4-Pack", "retail_price": 8.0, "product_category": "Confections", "product_sub_category": "Chocolate"}
#).execute()

#print("Inserted product:", insert_response.data)

# Update product name or price
update_response = supabase.table("product_master").update(
    {"retail_price": 9.0}
).eq("product_id", 201).execute()

print("Updated product:", update_response.data)