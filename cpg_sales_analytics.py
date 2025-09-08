# Sample CPG product
product_name = "Hershey's Milk Chocolate Bar"
price = 10.0           # USD
units_sold_last_week = 120
is_promoted = True

print(f"Product: {product_name}, Price: ${price}, Sold: {units_sold_last_week}, On promotion? {is_promoted}")

# List of products
products = ["Hershey's Milk Chocolate", "Oreo Cookies", "Coca-Cola Can"]
print("Products:", products)

# Add a new product
products.append("Pepsi Can")
print("Updated Products:", products)


# Dictionary storing product info
product_info = {
    "Hershey": {"price": 10.0, "units_sold": 120, "on_promo": True},
    "Oreo": {"price": 5.0, "units_sold": 200, "on_promo": False},
    "Coke": {"price": 2.0, "units_sold": 500, "on_promo": True}
}

# Accessing info
print("Oreo units sold last week:", product_info["Oreo"]["units_sold"])

# Total revenue for each product
for product, info in product_info.items():
    revenue = info["price"] * info["units_sold"]
    print(f"{product} generated ${revenue} last week")

# Use function for all products
def calculate_revenue(price, units_sold):
    return price * units_sold


for product, info in product_info.items():
    revenue = calculate_revenue(info["price"], info["units_sold"])
    print(f"{product} revenue (via function): ${revenue}")


# Simulate adding a new product dynamically
#new_product = input("Enter a new product name: ")
#new_units = int(input("Enter units sold last week: "))
#new_price = float(input("Enter price: "))

# Add to dictionary
#product_info[new_product] = {"price": new_price, "units_sold": new_units, "on_promo": False}
#print("Updated product info:", product_info)

# Calculate total sales across all products
def calculate_total_sales(product_data):
    total_sales = 0
    for info in product_data.values():
        total_sales += info["price"] * info["units_sold"]
    return total_sales

total_sales = calculate_total_sales(product_info)
print(f"Total sales across all products: ${total_sales}")

# Identify top-selling product
def top_selling_product(product_data):
    top_product = None
    max_units = 0
    for product, info in product_data.items():
        if info["units_sold"] > max_units:
            max_units = info["units_sold"]
            top_product = product
    return top_product, max_units

top_product, units = top_selling_product(product_info)
print(f"Top-selling product: {top_product} with {units} units sold")

# Products on promotion
def products_on_promotion(product_data):
    promo_products = []
    for product, info in product_data.items():
        if info["units_sold"] < 100:
            promo_products.append(product)
    return promo_products

promo_list = products_on_promotion(product_info)
print("Products needing promotion:", promo_list)