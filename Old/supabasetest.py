from supabase import create_client

url = "https://sywxhehahunevputgxdd.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"
supabase = create_client(url, key)

# Fetch all rows from product_master
response = supabase.table("product_master").select("*").execute()

print("Data:", response.data)
print("Count:", response.count)



# 1. INSERT a new product
print("=== INSERT ===")
insert_response = supabase.table("product_master").insert(
    {"product_id": 112, "product_name": "Oreo Double Stuf Cookies"}
).execute()
print("Inserted:", insert_response.data)

# 2. READ all products
print("\n=== READ ===")
read_response = supabase.table("product_master").select("*", count="exact").execute()
print("Data:", read_response.data)
print("Row count:", read_response.count)

# 3. UPDATE product name for id=111
print("\n=== UPDATE ===")
update_response = supabase.table("product_master").update(
    {"product_name": "Hershey's Dark Chocolate Bar 10Oz"}
).eq("product_id", 111).execute()
print("Updated:", update_response.data)

# 4. DELETE product with id=112
print("\n=== DELETE ===")
delete_response = supabase.table("product_master").delete().eq("product_id", 112).execute()
print("Deleted:", delete_response.data)

# 5. READ again to confirm changes
print("\n=== FINAL READ ===")
final_response = supabase.table("product_master").select("*", count="exact").execute()
print("Data:", final_response.data)
print("Row count:", final_response.count)