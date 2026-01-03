from dotenv import load_dotenv
import os

load_dotenv()
from db import engine, Base



def main():
    print("Creating tables (checkfirst=True)...")
    Base.metadata.create_all(bind=engine, checkfirst=True)
    print("Done.")

if __name__ == "__main__":
    main()
