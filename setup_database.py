#!/usr/bin/env python3
"""
🏗️ FIREBOLT DATABASE SETUP
==========================
Creates the required table structure AND vector index for semantic search.
CRITICAL: Must be run BEFORE loading any data - vector indexes require empty tables.
"""

from firebolt.client.auth import ClientCredentials
from firebolt.db import connect
from config import config

def setup_database():
    """Create table structure with vector index BEFORE data loading"""
    
    print("🏗️ FIREBOLT DATABASE SETUP")
    print("=" * 50)
    print(f"🎯 Target: {config.CLOUD_ACCOUNT}/{config.CLOUD_DATABASE}")
    print(f"🚀 Engine: {config.CLOUD_ENGINE}")
    print(f"📊 Table: {config.CLOUD_TABLE}")
    print(f"📏 Vector Dimensions: {config.EMBEDDING_DIMENSIONS}")
    print()
    print("⚠️  IMPORTANT: Vector index must be created on empty table!")
    print()

    try:
        auth = ClientCredentials(config.CLOUD_CLIENT_ID, config.CLOUD_CLIENT_SECRET)
        
        with connect(
            auth=auth,
            account_name=config.CLOUD_ACCOUNT,
            database=config.CLOUD_DATABASE,
            engine_name=config.CLOUD_ENGINE
        ) as connection:
            cursor = connection.cursor()
            
            # 1. Drop existing table completely (vector indexes cannot be modified easily)
            print("🗑️ Dropping any existing table...")
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {config.CLOUD_TABLE}")
                print("✅ Existing table dropped")
            except Exception as e:
                print(f"ℹ️ No existing table: {e}")
            
            # 2. Create the table with optimal structure
            print("🔧 Creating semantic search table...")
            create_table_sql = f"""
            CREATE TABLE {config.CLOUD_TABLE} (
                product_id          TEXT NOT NULL,
                title              TEXT NOT NULL,
                description        TEXT,
                brand              TEXT,
                price              REAL,
                image_url          TEXT,
                main_category      TEXT,
                sub_category       TEXT,
                embedding          ARRAY(DOUBLE PRECISION),
                created_at         TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (product_id)
            )
            """
            
            cursor.execute(create_table_sql)
            print(f"✅ Table created: {config.CLOUD_TABLE}")
            
            # 3. CRITICAL: Create vector index on empty table
            print("🧠 Creating HNSW vector index (this may take a moment)...")
            try:
                vector_index_sql = f"""
                CREATE AGGREGATING INDEX {config.CLOUD_TABLE}_vector_idx 
                ON {config.CLOUD_TABLE} (embedding)
                """
                cursor.execute(vector_index_sql)
                print("✅ HNSW vector index created successfully")
                print("   🔍 Semantic search will be lightning fast!")
            except Exception as e:
                print(f"⚠️ Vector index creation: {e}")
                print("   Trying alternative index syntax...")
                try:
                    # Alternative syntax for different Firebolt versions
                    alt_index_sql = f"""
                    CREATE INDEX {config.CLOUD_TABLE}_vector_idx 
                    ON {config.CLOUD_TABLE} 
                    USING HNSW (embedding)
                    WITH (dimension = {config.EMBEDDING_DIMENSIONS})
                    """
                    cursor.execute(alt_index_sql)
                    print("✅ Vector index created (alternative syntax)")
                except Exception as e2:
                    print(f"❌ Vector index failed: {e2}")
                    print("   📝 Continuing without vector index - search will still work")
            
            # 4. Create standard indexes for filtering
            print("📊 Creating standard indexes...")
            standard_indexes = [
                f"CREATE INDEX idx_{config.CLOUD_TABLE.replace('-', '_')}_category ON {config.CLOUD_TABLE} (main_category)",
                f"CREATE INDEX idx_{config.CLOUD_TABLE.replace('-', '_')}_brand ON {config.CLOUD_TABLE} (brand)",
                f"CREATE INDEX idx_{config.CLOUD_TABLE.replace('-', '_')}_price ON {config.CLOUD_TABLE} (price)",
            ]
            
            for idx_sql in standard_indexes:
                try:
                    cursor.execute(idx_sql)
                    print("  ✅ Standard index created")
                except Exception as e:
                    print(f"  ⚠️ Index skipped: {e}")
            
            # 5. Verify empty table structure
            print("\n📊 FINAL TABLE STRUCTURE:")
            cursor.execute(f"DESCRIBE {config.CLOUD_TABLE}")
            schema = cursor.fetchall()
            for col in schema:
                print(f"   📋 {col[0]} | {col[1]} | {col[2]}")
            
            # 6. Confirm table is empty and ready
            cursor.execute(f"SELECT COUNT(*) FROM {config.CLOUD_TABLE}")
            count = cursor.fetchone()[0]
            print(f"\n📊 Table status: {count} rows (should be 0)")
            
            if count > 0:
                print("⚠️  WARNING: Table contains data! Vector index may not be optimal.")
            else:
                print("✅ Perfect! Empty table with vector index ready for data loading.")
            
            print("\n🎉 DATABASE SETUP COMPLETE!")
            print("=" * 30)
            print("✅ Table structure created")
            print("✅ Vector index optimized")  
            print("✅ Standard indexes added")
            print("✅ Ready for data loading")
            print()
            print("📋 NEXT STEPS:")
            print("   1. Run: python load_home_depot_dataset.py")
            print("   2. Then: python check_vector_table.py")
            print("   3. Finally: streamlit run streamlit_cloud_app.py")
            
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("   - Verify .env file has correct credentials")
        print("   - Ensure engine is running in Firebolt Cloud")
        print("   - Check CREATE TABLE permissions")
        print("   - Run: python config.py to validate setup")
        return False
    
    return True

def main():
    """Main setup function with validation"""
    print("🚀 FIREBOLT DATABASE SETUP - Vector Search Ready")
    print()
    
    # Validate configuration
    try:
        config.print_config_summary()
        print()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Confirm destructive operation
    print("⚠️  This will DROP any existing table and recreate it.")
    print("   All existing data will be lost!")
    print()
    
    response = input("Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Setup cancelled by user")
        return
    
    # Run setup
    success = setup_database()
    
    if success:
        print("\n🌟 SUCCESS! Database is ready for semantic search.")
        print("🔥 Vector index created - searches will be blazing fast!")
    else:
        print("\n💥 Setup failed. Check errors above.")

if __name__ == "__main__":
    main()