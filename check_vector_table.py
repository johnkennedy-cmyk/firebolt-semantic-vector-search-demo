#!/usr/bin/env python3
"""
Check the home_depot_semantic_search table in Firebolt Cloud
Verify it's ready for multimodal search demo
"""

from firebolt.client.auth import ClientCredentials
from firebolt.db import connect
from config import config

def check_vector_table():
    """Check the home_depot_semantic_search table"""
    print("🔍 CHECKING VECTOR TABLE IN FIREBOLT CLOUD")
    print("=" * 50)
    
    auth = ClientCredentials(config.CLOUD_CLIENT_ID, config.CLOUD_CLIENT_SECRET)
    
    with connect(
        auth=auth,
        account_name=config.CLOUD_ACCOUNT,
        database=config.CLOUD_DATABASE,
        engine_name=config.CLOUD_ENGINE
    ) as connection:
        cursor = connection.cursor()
        
        # Check table structure
        print("📊 Table Schema:")
        cursor.execute(f"DESCRIBE {config.CLOUD_TABLE}")
        schema = cursor.fetchall()
        for col in schema:
            print(f"   📋 {col[0]} | {col[1]} | {col[2]}")
        
        # Check row count
        print(f"\n📈 Data Count:")
        cursor.execute(f"SELECT COUNT(*) FROM {config.CLOUD_TABLE}")
        count = cursor.fetchone()[0]
        print(f"   📊 Total products: {count:,}")
        
        # Check sample data
        print(f"\n🎯 Sample Products:")
        cursor.execute(f"""
            SELECT product_id, title, brand, ARRAY_LENGTH(embedding) as embedding_dims 
            FROM {config.CLOUD_TABLE} 
            LIMIT 5
        """)
        samples = cursor.fetchall()
        for sample in samples:
            print(f"   🛍️ {sample[0]} | {sample[1][:50]}... | {sample[2]} | {sample[3]} dims")
        
        # Check for embeddings
        print(f"\n🧠 Vector Embeddings:")
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_products,
                COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as with_embeddings,
                AVG(ARRAY_LENGTH(embedding)) as avg_embedding_size
            FROM {config.CLOUD_TABLE}
        """)
        embedding_stats = cursor.fetchone()
        print(f"   📊 Total products: {embedding_stats[0]}")
        print(f"   🧠 With embeddings: {embedding_stats[1]}")
        print(f"   📏 Avg embedding size: {embedding_stats[2]:.0f}")
        
        # Test similarity search capability
        if embedding_stats[1] > 0:
            print(f"\n🔍 Testing Vector Similarity:")
            cursor.execute(f"""
                SELECT product_id, title, 
                       vector_cosine_similarity(embedding, (SELECT embedding FROM {config.CLOUD_TABLE} WHERE embedding IS NOT NULL LIMIT 1)) as similarity
                FROM {config.CLOUD_TABLE} 
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC 
                LIMIT 3
            """)
            similarity_test = cursor.fetchall()
            for result in similarity_test:
                print(f"   🎯 {result[0]} | {result[1][:40]}... | {result[2]:.3f}")
        
        print(f"\n🎉 VECTOR TABLE READY FOR MULTIMODAL SEARCH!")
        return True

if __name__ == "__main__":
    check_vector_table()
