import os
import pymysql
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "olist")

def run_sql_file(filename, connection):
    print(f"Executing {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        # Split by semicolon, but be careful with complex queries
        # For our scripts, simple splitting works
        sql_commands = f.read().split(';')
        
        with connection.cursor() as cursor:
            for command in sql_commands:
                if command.strip():
                    try:
                        cursor.execute(command)
                    except Exception as e:
                        print(f"Warning/Error in {filename}: {e}")
            connection.commit()

def setup():
    # 1. Connect to MySQL (Initially without a specific database to run schema.sql)
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=int(DB_PORT),
            local_infile=True, # Enable client-side local infile
            autocommit=True
        )
    except Exception as e:
        print(f"CRITICAL ERROR: Could not connect to MySQL. Check your .env file credentials.\n{e}")
        return

    try:
        # 2. Enable server-side local_infile (requires SUPER privilege)
        with conn.cursor() as cursor:
            print("Enabling local_infile on server...")
            cursor.execute("SET GLOBAL local_infile = 1;")
        
        # 3. Run Schema
        run_sql_file('sql/schema.sql', conn)
        
        # 4. Run Data Loading
        run_sql_file('sql/load_data.sql', conn)
        
        # 5. Run Feature Queries (Views)
        run_sql_file('sql/feature_queries.sql', conn)
        
        print("\n✅ Database setup complete! You can now explore 'analytical_dataset' in your VS Code extension.")
        
    except Exception as e:
        print(f"An error occurred during setup: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    setup()
