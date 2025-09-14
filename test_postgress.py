import psycopg2
import os
import socket
from dotenv import load_dotenv

load_dotenv()

# Force IPv4 resolution
host = os.getenv("DB_HOST")
ipv4 = socket.gethostbyname(host)

print("Using IPv4:", ipv4)

try:
    conn = psycopg2.connect(
        host=ipv4,  # Use IPv4 directly
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
        sslmode="require"
    )
    print("✅ Connection successful!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:", e)