#!/usr/bin/env python3
"""
Load Home Depot Dataset into Firebolt Cloud
Smart batch processing with embedding generation for 2,551+ products
"""

import pandas as pd
import numpy as np
import json
import time
import requests
from typing import List, Dict, Any
from pathlib import Path
from firebolt.client.auth import ClientCredentials
from firebolt.db import connect
from config import config

# Dataset Configuration
CSV_PATH = Path.home() / "Downloads" / "home_depot_data_1_2021_12.csv"
BATCH_SIZE = 50  # Process in batches to avoid memory issues
MAX_PRODUCTS = 500  # Limit for initial load (can be increased)

class HomeDepotLoader:
    def __init__(self):
        self.auth = ClientCredentials(config.CLOUD_CLIENT_ID, config.CLOUD_CLIENT_SECRET)
        self.processed_count = 0
        self.embedding_service_url = config.OLLAMA_URL
        
    def get_connection(self):
        """Get Firebolt Cloud connection"""
        return connect(
            auth=self.auth,
            account_name=config.CLOUD_ACCOUNT,
            database=config.CLOUD_DATABASE,
            engine_name=config.CLOUD_ENGINE
        )
    
    def check_ollama_service(self):
        """Check if Ollama is running for embedding generation"""
        try:
            response = requests.get(f"{self.embedding_service_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                has_model = any(config.EMBEDDING_MODEL in model.get("name", "") for model in models)
                print(f"✅ Ollama running with {len(models)} models")
                if has_model:
                    print(f"✅ {config.EMBEDDING_MODEL} model available")
                    return True
                else:
                    print(f"⚠️ {config.EMBEDDING_MODEL} model not found")
                    return False
            return False
        except Exception as e:
            print(f"❌ Ollama not accessible: {e}")
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama model"""
        try:
            response = requests.post(
                f"{self.embedding_service_url}/api/embeddings",
                json={
                    "model": config.EMBEDDING_MODEL,
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                if len(embedding) == config.EMBEDDING_DIMENSIONS:
                    return embedding
                else:
                    print(f"⚠️ Unexpected embedding size: {len(embedding)}")
                    return []
            else:
                print(f"❌ Embedding request failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Embedding generation error: {e}")
            return []
    
    def clean_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the raw dataset"""
        print("🧹 Cleaning and preparing data...")
        
        # Drop rows with missing essential data
        df_clean = df.dropna(subset=['title', 'description']).copy()
        
        # Clean price column
        df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce')
        df_clean = df_clean[df_clean['price'] > 0]  # Remove invalid prices
        
        # Clean brand column
        df_clean['brand'] = df_clean['brand'].fillna('Unknown')
        
        # Create combined text for embedding generation
        df_clean['combined_text'] = (
            df_clean['title'].astype(str) + ". " + 
            df_clean['description'].astype(str) + ". " +
            "Brand: " + df_clean['brand'].astype(str)
        )
        
        # Extract first image URL
        df_clean['image_url'] = df_clean['images'].apply(self.extract_first_image)
        
        # Create categories from title/description analysis
        df_clean['main_category'] = df_clean['title'].apply(self.categorize_product)
        df_clean['sub_category'] = df_clean['description'].apply(lambda x: str(x)[:100] + "...")
        
        print(f"✅ Cleaned data: {len(df_clean)} valid products")
        return df_clean
    
    def extract_first_image(self, images_str: str) -> str:
        """Extract the first image URL from the images string"""
        if pd.isna(images_str) or not images_str:
            return ""
        
        # Split by ~ and take first URL
        urls = str(images_str).split("~")
        return urls[0].strip() if urls else ""
    
    def categorize_product(self, title: str) -> str:
        """Simple categorization based on title keywords"""
        title_lower = str(title).lower()
        
        categories = {
            'Tools': ['drill', 'saw', 'hammer', 'screwdriver', 'wrench'],
            'Appliances': ['refrigerator', 'dishwasher', 'microwave', 'oven', 'washer'],
            'Lighting': ['light', 'lamp', 'bulb', 'chandelier', 'fixture'],
            'Plumbing': ['sink', 'faucet', 'toilet', 'pipe', 'valve'],
            'Electrical': ['wire', 'outlet', 'switch', 'cable', 'adapter'],
            'Hardware': ['screw', 'nail', 'bolt', 'hinge', 'bracket'],
            'Garden': ['plant', 'seed', 'fertilizer', 'hose', 'mower'],
            'Clothing': ['shirt', 'pants', 'jacket', 'boot', 'glove']
        }
        
        for category, keywords in categories.items():
            if any(keyword in title_lower for keyword in keywords):
                return category
        
        return 'General'
    
    def clear_existing_data(self):
        """Clear existing data in the table"""
        print("🗑️ Clearing existing data...")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {config.CLOUD_TABLE}")
            print("✅ Existing data cleared")
    
    def load_batch_to_firebolt(self, batch_df: pd.DataFrame):
        """Load a batch of products to Firebolt"""
        print(f"📤 Loading batch of {len(batch_df)} products to Firebolt Cloud...")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in batch_df.iterrows():
                try:
                    # Generate embedding
                    print(f"🧠 Generating embedding for: {row['title'][:50]}...")
                    embedding = self.generate_embedding(row['combined_text'])
                    
                    if not embedding:
                        print(f"⚠️ Skipping product due to embedding failure")
                        continue
                    
                    # Insert into database
                    insert_sql = f"""
                    INSERT INTO {config.CLOUD_TABLE} (
                        product_id, title, description, brand, price, image_url,
                        main_category, sub_category, embedding, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
                    """
                    
                    cursor.execute(insert_sql, (
                        str(row['product_id']),
                        str(row['title'])[:500],  # Limit length
                        str(row['description'])[:1000],
                        str(row['brand'])[:100],
                        float(row['price']),
                        str(row['image_url'])[:500],
                        str(row['main_category']),
                        str(row['sub_category'])[:200],
                        embedding
                    ))
                    
                    self.processed_count += 1
                    print(f"✅ Loaded product {self.processed_count}: {row['title'][:30]}...")
                    
                    # Small delay to avoid overwhelming the system
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ Error loading product {row['product_id']}: {e}")
                    continue
    
    def load_dataset(self):
        """Main loading function"""
        print("🚀 STARTING HOME DEPOT DATASET LOADING")
        print("=" * 50)
        print(f"📊 Target: {config.CLOUD_ACCOUNT}/{config.CLOUD_DATABASE}/{config.CLOUD_TABLE}")
        print(f"📁 Source: {CSV_PATH}")
        print(f"⚙️ Batch Size: {BATCH_SIZE}")
        print(f"🔢 Max Products: {MAX_PRODUCTS}")
        print()
        
        # Check prerequisites
        if not CSV_PATH.exists():
            print(f"❌ CSV file not found: {CSV_PATH}")
            return False
        
        if not self.check_ollama_service():
            print("❌ Ollama service required for embedding generation")
            return False
        
        # Load and prepare data
        print("📖 Loading CSV data...")
        df = pd.read_csv(CSV_PATH)
        print(f"✅ Loaded {len(df)} raw records")
        
        # Clean data
        df_clean = self.clean_and_prepare_data(df)
        
        # Limit dataset size for initial load
        if len(df_clean) > MAX_PRODUCTS:
            print(f"⚠️ Limiting to first {MAX_PRODUCTS} products for initial load")
            df_clean = df_clean.head(MAX_PRODUCTS)
        
        # Clear existing data
        self.clear_existing_data()
        
        # Process in batches
        print(f"🔄 Processing {len(df_clean)} products in batches of {BATCH_SIZE}...")
        
        start_time = time.time()
        
        for i in range(0, len(df_clean), BATCH_SIZE):
            batch = df_clean.iloc[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(df_clean) - 1) // BATCH_SIZE + 1
            
            print(f"\n📦 BATCH {batch_num}/{total_batches}")
            print("-" * 30)
            
            self.load_batch_to_firebolt(batch)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time_per_product = elapsed / self.processed_count if self.processed_count > 0 else 0
            remaining_products = len(df_clean) - self.processed_count
            estimated_remaining = remaining_products * avg_time_per_product
            
            print(f"📈 Progress: {self.processed_count}/{len(df_clean)} products")
            print(f"⏱️ Avg time per product: {avg_time_per_product:.2f}s")
            print(f"🕐 Estimated remaining: {estimated_remaining/60:.1f} minutes")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎉 LOADING COMPLETE!")
        print("=" * 20)
        print(f"✅ Products loaded: {self.processed_count}")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📊 Average: {total_time/self.processed_count:.2f}s per product")
        
        return True

def main():
    """Main function"""
    loader = HomeDepotLoader()
    success = loader.load_dataset()
    
    if success:
        print("\n🌟 Dataset ready for semantic search!")
        print("🔍 Try searching for:")
        print("   - 'power tools for home improvement'")
        print("   - 'kitchen appliances and fixtures'") 
        print("   - 'outdoor garden equipment'")
        print("   - 'electrical wiring supplies'")
    else:
        print("\n❌ Loading failed - check logs for details")

if __name__ == "__main__":
    main()
