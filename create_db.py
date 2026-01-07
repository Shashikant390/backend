from dotenv import load_dotenv
import os
from sqlalchemy import text  # <--- Import 'text' to run raw SQL

load_dotenv()
from db import engine, Base
import models 

def main():
    print("--- Database Initialization ---")
    print(f"Target: {engine.url}")

    # ---------------------------------------------------------
    # 1. Enable PostGIS (The Fix)
    # ---------------------------------------------------------
    print("🔧 Enabling PostGIS extension...")
    with engine.connect() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        connection.commit()  # Save the change
    print("✅ PostGIS enabled.")

    # ---------------------------------------------------------
    # 2. Create Tables
    # ---------------------------------------------------------
    print("Creating tables...")
    Base.metadata.create_all(bind=engine, checkfirst=True)
    print("✅ Done. All tables created successfully.")

if __name__ == "__main__":
    main()