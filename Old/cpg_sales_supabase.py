from supabase import create_client, Client

url = "https://sywxhehahunevputgxdd.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3hoZWhhaHVuZXZwdXRneGRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTczMDQxNDEsImV4cCI6MjA3Mjg4MDE0MX0.SGNrmklPobim7-zb4zs78e2i2VRrmV84gV7DIl2m5s8"

supabase = create_client(url, key)

# -------------------------------
# 2️⃣ Product Master Operations
# -------------------------------
def add_product(name, category, sub_category, price):
    """Add a new product to product_master"""
    response = supabase.table("product_master").insert(
        {
            "product_name": name,
            "product_category": category,
            "product_sub_category": sub_category,
            "retail_price": price
        }
    ).execute()
    return response.data


def get_all_products():
    """Fetch all products"""
    return supabase.table("product_master").select("*").execute().data



# -------------------------------
# 3️⃣ Sales Transactions Operations
# -------------------------------
def add_sale(product_id, units_sold, sale_price, store_id=None, region=None):
    """Insert a new sales transaction"""
    response = supabase.table("sales_transactions").insert(
        {
            "product_id": product_id,
            "units_sold": units_sold,
            "sale_price": sale_price,
            "store_id": store_id,
            "region": region
        }
    ).execute()
    return response.data

def get_sales_by_product(product_id):
    """Fetch sales for a specific product"""
    return supabase.table("sales_transactions").select("*").eq("product_id", product_id).execute().data



# -------------------------------
# 4️⃣ Analytics Functions
# -------------------------------
def calculate_total_revenue(product_id):
    """Calculate total revenue for a product from sales_transactions"""
    sales = get_sales_by_product(product_id)
    total_revenue = sum(s["units_sold"] * s["sale_price"] for s in sales)
    return total_revenue


def show_revenue_report():
    products = get_all_products()
    print("\n=== Revenue Report ===")
    for product in products:
        revenue = calculate_total_revenue(product["product_id"])
        print(f"{product['product_name']} | Total Revenue: ${revenue}")


# -------------------------------
# 5️⃣ Interactive Menu
# -------------------------------
def main_menu():
    while True:
        print("\n=== CPG Management Menu ===")
        print("1. Add Product")
        print("2. Record Sale")
        print("3. Show Revenue Report")
        print("4. List Products")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            name = input("Product Name: ")
            category = input("Category: ")
            sub_category = input("Sub-Category: ")
            price = float(input("Retail Price: "))
            add_product(name, category, sub_category, price)

        elif choice == "2":
            products = get_all_products()
            print("\nAvailable Products:")
            for p in products:
                print(f"{p['product_id']}: {p['product_name']}")
            product_id = int(input("Enter Product ID to record sale: "))
            units_sold = int(input("Units Sold: "))
            sale_price = float(input("Sale Price: "))
            store_id = input("Store ID (optional): ") or None
            region = input("Region (optional): ") or None
            add_sale(product_id, units_sold, sale_price, store_id, region)

        elif choice == "3":
            show_revenue_report()

        elif choice == "4":
            products = get_all_products()
            print("\n=== Product List ===")
            for p in products:
                print(f"{p['product_id']} | {p['product_name']} | {p['product_category']} | ${p['retail_price']}")

        elif choice == "5":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Try again.")

# -------------------------------
# Run Interactive Menu
# -------------------------------
if __name__ == "__main__":
    main_menu()