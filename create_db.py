from dotenv import load_dotenv
import os

load_dotenv()
from db import engine, Base

# ---------------------------------------------------------
# CRITICAL FIX: You MUST import models here.
# Even though you don't use 'models' directly, importing it
# registers the classes (AppUser, Farm, etc.) with Base.
# ---------------------------------------------------------
import models 

def main():
    print("Checking database connection...")
    print(f"Target Database: {engine.url}")
    
    # This line checks if SQLAlchemy actually 'sees' your tables
    detected_tables = Base.metadata.tables.keys()
    print(f"Tables detected in code: {list(detected_tables)}")

    if not detected_tables:
        print("❌ ERROR: No tables detected! Make sure 'import models' is present.")
        return

    print("Creating tables (checkfirst=True)...")
    Base.metadata.create_all(bind=engine, checkfirst=True)
    print("✅ Done. Tables created successfully.")

if __name__ == "__main__":
    main()