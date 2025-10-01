#!/usr/bin/env python3
"""
🚀 FIREBOLT CLOUD SEMANTIC SEARCH DEMO
======================================
Cloud-enabled version using Firebolt Cloud with HNSW vector indexes
"""

import streamlit as st
import pandas as pd
import json
import time
import requests
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from firebolt.db import connect
from firebolt.client.auth import ClientCredentials
from config import config

# 🎨 Page Configuration
st.set_page_config(
    page_title="🔍 Firebolt Cloud Semantic Search",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CloudSemanticSearch:
    def __init__(self):
        """Initialize cloud connection to Firebolt"""
        # 🔒 Secure Configuration from Environment Variables
        self.account_name = config.CLOUD_ACCOUNT
        self.engine_name = config.CLOUD_ENGINE
        self.database = config.CLOUD_DATABASE
        self.table_name = config.CLOUD_TABLE
        
        # Initialize connection
        self.connection = None
        self.search_history = []
        
    def get_connection(self):
        """Get or create Firebolt cloud connection"""
        if self.connection is None:
            try:
                # 🔒 Secure credentials from environment variables
                auth = ClientCredentials(
                    client_id=config.CLOUD_CLIENT_ID,
                    client_secret=config.CLOUD_CLIENT_SECRET
                )
                
                self.connection = connect(
                    engine_name=self.engine_name,
                    database=self.database,
                    account_name=self.account_name,
                    auth=auth
                )
                return self.connection
            except Exception as e:
                st.error(f"❌ Cloud Connection Failed: {e}")
                return None
        return self.connection

    def query_firebolt(self, sql: str) -> Optional[Dict]:
        """Execute SQL query on Firebolt Cloud"""
        try:
            connection = self.get_connection()
            if not connection:
                return None
                
            cursor = connection.cursor()
            cursor.execute(sql)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Fetch all results
            rows = cursor.fetchall()
            
            return {
                'columns': columns,
                'data': rows
            }
            
        except Exception as e:
            st.error(f"❌ Query Failed: {e}")
            return None

    def get_local_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using local Ollama"""
        try:
            response = requests.post(
                f"{config.OLLAMA_URL}/api/embeddings",
                json={"model": config.EMBEDDING_MODEL, "prompt": text},
                timeout=15
            )
            
            if response.status_code == 200:
                embedding = response.json().get('embedding', [])
                if len(embedding) == config.EMBEDDING_DIMENSIONS:
                    return embedding
                    
        except Exception as e:
            st.error(f"❌ Embedding Error: {e}")
            
        return None

    def semantic_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """🚀 CLOUD SEMANTIC SEARCH with HNSW Vector Index"""
        start_time = time.time()
        
        # Generate query embedding
        embedding_start = time.time()
        query_embedding = self.get_local_embedding(query)
        embedding_time = time.time() - embedding_start
        
        if not query_embedding:
            return {
                'results': [],
                'query_embedding': None,
                'query': query,
                'performance': {
                    'total_time': time.time() - start_time,
                    'embedding_time': embedding_time,
                    'db_time': 0,
                    'num_results': 0,
                    'method': 'embedding_failed'
                }
            }
        
        # Convert embedding to SQL format
        embedding_str = json.dumps(query_embedding)
        
        # 🚀 FIREBOLT CLOUD NATIVE VECTOR SEARCH - Direct table access
        db_start = time.time()
        
        # Try different table name variations since SDK has visibility issues
        possible_table_names = [
            self.table_name,  # home_depot_semantic_search
            f'"{self.table_name}"',  # "home_depot_semantic_search" 
            f'`{self.table_name}`',  # `home_depot_semantic_search`
            self.table_name.upper(),  # HOME_DEPOT_SEMANTIC_SEARCH
            self.table_name.lower()   # home_depot_semantic_search
        ]
        
        sql_template = """
        SELECT 
            product_id, 
            title, 
            description, 
            price, 
            brand, 
            main_category,
            embedding,
            vector_cosine_similarity(
                embedding, 
                CAST({embedding_str} AS ARRAY(FLOAT))
            ) as similarity_score
        FROM {table_name} 
        WHERE embedding IS NOT NULL
        ORDER BY similarity_score DESC
        LIMIT {limit}
        """
        
        result = None
        successful_table_name = None
        
        # Try each table name variation
        for table_name in possible_table_names:
            sql = sql_template.format(
                embedding_str=embedding_str,
                table_name=table_name,
                limit=limit
            )
            
            try:
                result = self.query_firebolt(sql)
                if result and 'data' in result and result['data']:
                    successful_table_name = table_name
                    st.sidebar.success(f"✅ Successfully accessed table as: {table_name}")
                    break
                elif result and 'data' in result:
                    # Query worked but no data - still success
                    successful_table_name = table_name
                    st.sidebar.info(f"✅ Table accessible as: {table_name} (but no data returned)")
                    break
            except Exception as e:
                st.sidebar.warning(f"❌ Failed with table name '{table_name}': {str(e)[:50]}...")
                continue
        
        if not successful_table_name:
            st.sidebar.error("❌ Could not access table with any name variation")
            sql = sql_template.format(
                embedding_str=embedding_str,
                table_name=self.table_name,
                limit=limit
            )
        
        result = self.query_firebolt(sql)
        db_time = time.time() - db_start
        
        if not result or 'data' not in result:
            return {
                'results': [],
                'query_embedding': query_embedding,
                'query': query,
                'performance': {
                    'total_time': time.time() - start_time,
                    'embedding_time': embedding_time,
                    'db_time': db_time,
                    'num_results': 0,
                    'method': 'firebolt_cloud_hnsw'
                }
            }
        
        # Format results
        results = []
        for row in result['data']:
            results.append({
                'product_id': row[0],
                'title': row[1],
                'description': row[2] or "",
                'price': float(row[3]) if row[3] else 0,
                'brand': row[4] or "",
                'category': row[5] or "",
                'similarity': float(row[7]),
                'search_method': 'firebolt_cloud_hnsw',
                'product_embedding': row[6]
            })
        
        # Track performance
        total_time = time.time() - start_time
        performance_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'query': query,
            'total_time': total_time,
            'embedding_time': embedding_time,
            'db_time': db_time,
            'num_results': len(results),
            'method': 'firebolt_cloud_hnsw'
        }
        
        # Add to search history
        self.search_history.append(performance_data)
        if len(self.search_history) > 100:
            self.search_history.pop(0)
        
        return {
            'results': results,
            'query_embedding': query_embedding,
            'query': query,
            'performance': performance_data
        }

    def get_product_stats(self) -> Dict:
        """Get product statistics from cloud database"""
        # Skip SHOW TABLES since it fails - try direct access with different name variations
        possible_table_names = [
            self.table_name,  # home_depot_semantic_search
            f'"{self.table_name}"',  # "home_depot_semantic_search" 
            f'`{self.table_name}`',  # `home_depot_semantic_search`
            self.table_name.upper(),  # HOME_DEPOT_SEMANTIC_SEARCH
            self.table_name.lower()   # home_depot_semantic_search
        ]
        
        sql_template = """
        SELECT 
            COUNT(*) as total_products,
            COUNT(DISTINCT brand) as unique_brands,
            COUNT(DISTINCT main_category) as unique_categories,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price
        FROM {table_name}
        """
        
        # Try each table name variation
        for table_name in possible_table_names:
            sql = sql_template.format(table_name=table_name)
            
            try:
                result = self.query_firebolt(sql)
                if result and result['data']:
                    row = result['data'][0]
                    # Success! Store which table name worked
                    self.working_table_name = table_name
                    return {
                        'total_products': int(row[0]),
                        'unique_brands': int(row[1]),
                        'unique_categories': int(row[2]),
                        'avg_price': float(row[3]) if row[3] else 0,
                        'min_price': float(row[4]) if row[4] else 0,
                        'max_price': float(row[5]) if row[5] else 0,
                        'working_table_name': table_name
                    }
            except Exception as e:
                continue
        
        return {'error': 'Could not access table with any name variation'}

    def add_new_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new product to the database with automatic embedding generation"""
        try:
            # Generate combined text for embedding
            combined_text = f"{product_data['title']}. {product_data['description']}. Brand: {product_data['brand']}"
            
            # Generate embedding
            embedding = self.get_local_embedding(combined_text)
            if not embedding:
                return {'success': False, 'error': 'Failed to generate embedding'}
            
            # Generate unique product ID
            import time
            import random
            product_id = f"CUSTOM_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # Insert into database
            insert_sql = """
            INSERT INTO home_depot_semantic_search (
                product_id, title, description, brand, price, image_url,
                main_category, sub_category, embedding, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
            """
            
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute(insert_sql, (
                product_id,
                product_data['title'][:500],
                product_data['description'][:1000], 
                product_data['brand'][:100],
                float(product_data['price']),
                product_data['image_url'][:500],
                product_data['main_category'],
                product_data['sub_category'][:200],
                embedding
            ))
            
            return {
                'success': True, 
                'product_id': product_id,
                'embedding_size': len(embedding),
                'message': f'Product {product_id} added successfully!'
            }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

def display_search_results_with_embeddings(search_results: Dict, show_embeddings: bool = False):
    """Display search results with optional embedding analysis"""
    results = search_results['results']
    query = search_results['query']
    query_embedding = search_results['query_embedding']
    performance = search_results['performance']
    
    if not results:
        st.warning("🤷‍♂️ No results found. Try a different search term!")
        return
    
    # 📊 Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔍 Results", len(results))
    with col2:
        st.metric("⚡ Total Time", f"{performance['total_time']:.3f}s")
    with col3:
        st.metric("🧠 Embedding Time", f"{performance['embedding_time']:.3f}s") 
    with col4:
        st.metric("💽 DB Time", f"{performance['db_time']:.3f}s")
    
    # 🎯 Search Results
    st.subheader(f"🎯 Search Results for: *{query}*")
    
    for i, product in enumerate(results, 1):
        with st.expander(f"#{i} {product['title']} - {product['brand']} (Similarity: {product['similarity']:.4f})", expanded=(i <= 3)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description:** {product['description']}")
                st.write(f"**Category:** {product['category']}")
                st.write(f"**Price:** ${product['price']:.2f}")
                st.write(f"**Search Method:** {product['search_method']}")
            
            with col2:
                # Similarity gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = product['similarity'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Similarity"},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)
    
    # 🧠 Embedding Analysis (EXPANDED VIEW)
    if show_embeddings and query_embedding:
        st.subheader("🧠 Vector Embedding Analysis")
        
        # Query embedding stats
        query_stats = {
            'min': min(query_embedding),
            'max': max(query_embedding),
            'mean': sum(query_embedding) / len(query_embedding),
            'dimensions': len(query_embedding)
        }
        
        # Full-width embedding display
        st.markdown("### 📝 Query Embedding Vector")
        
        # Stats in metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dimensions", query_stats['dimensions'])
        with col2:
            st.metric("Min Value", f"{query_stats['min']:.4f}")
        with col3:
            st.metric("Max Value", f"{query_stats['max']:.4f}")
        with col4:
            st.metric("Mean Value", f"{query_stats['mean']:.4f}")
        
        # Expanded embedding preview
        st.write("**First 50 dimensions:**")
        embedding_preview = [f"{val:.4f}" for val in query_embedding[:50]]
        st.code(f"[{', '.join(embedding_preview)}...]", language="python")
        
        # Visualization
        fig = go.Figure(data=go.Scatter(
            y=query_embedding[:100], 
            mode='lines+markers',
            name='Query Embedding (first 100 dims)',
            line=dict(color='blue')
        ))
        fig.update_layout(
            title="Query Embedding Visualization (First 100 Dimensions)",
            xaxis_title="Dimension Index",
            yaxis_title="Embedding Value",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Product embedding comparison
        if results:
            product_embedding = results[0]['product_embedding']
            if product_embedding:
                st.markdown("### 🛍️ Top Product Embedding Vector")
                
                product_stats = {
                    'min': min(product_embedding),
                    'max': max(product_embedding),
                    'mean': sum(product_embedding) / len(product_embedding),
                    'dimensions': len(product_embedding)
                }
                
                # Product stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Dimensions", product_stats['dimensions'])
                with col2:
                    st.metric("Min Value", f"{product_stats['min']:.4f}")
                with col3:
                    st.metric("Max Value", f"{product_stats['max']:.4f}")
                with col4:
                    st.metric("Mean Value", f"{product_stats['mean']:.4f}")
                
                # Product embedding preview
                st.write("**First 50 dimensions:**")
                product_preview = [f"{val:.4f}" for val in product_embedding[:50]]
                st.code(f"[{', '.join(product_preview)}...]", language="python")
                
                # Comparison visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=query_embedding[:100], 
                    mode='lines+markers',
                    name='Query Embedding',
                    line=dict(color='blue'),
                    opacity=0.7
                ))
                fig.add_trace(go.Scatter(
                    y=product_embedding[:100], 
                    mode='lines+markers',
                    name=f"Product: {results[0]['title'][:30]}...",
                    line=dict(color='red'),
                    opacity=0.7
                ))
                fig.update_layout(
                    title="Embedding Comparison: Query vs Top Product (First 100 Dimensions)",
                    xaxis_title="Dimension Index",
                    yaxis_title="Embedding Value",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Similarity calculation
                query_array = np.array(query_embedding)
                product_array = np.array(product_embedding)
                
                # Cosine similarity
                dot_product = np.dot(query_array, product_array)
                norms = np.linalg.norm(query_array) * np.linalg.norm(product_array)
                similarity = dot_product / norms
                
                st.metric("🎯 Cosine Similarity", f"{similarity:.6f}", 
                         help="Higher values (closer to 1.0) indicate better semantic match")

def main():
    """Main Streamlit application"""
    st.title("🚀 Firebolt Cloud Semantic Search Demo")
    st.markdown("---")
    
    # Initialize search engine
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = CloudSemanticSearch()
    
    search_engine = st.session_state.search_engine
    
    # 🌐 Connection Status
    st.sidebar.header("🌐 Cloud Connection")
    connection = search_engine.get_connection()
    if connection:
        st.sidebar.success("✅ Connected to Firebolt Cloud")
        st.sidebar.info(f"**Account:** {search_engine.account_name}")
        st.sidebar.info(f"**Database:** {search_engine.database}")
        st.sidebar.info(f"**Engine:** {search_engine.engine_name}")
        st.sidebar.info(f"**Table:** {search_engine.table_name}")
    else:
        st.sidebar.error("❌ Cloud Connection Failed")
        st.error("Cannot connect to Firebolt Cloud. Please check your connection.")
        return
    
    # 📊 Database Stats
    st.sidebar.header("📊 Database Stats")
    try:
        stats = search_engine.get_product_stats()
        if stats:
            st.sidebar.metric("Products", stats.get('total_products', 0))
            st.sidebar.metric("Brands", stats.get('unique_brands', 0))
            st.sidebar.metric("Categories", stats.get('unique_categories', 0))
            st.sidebar.metric("Avg Price", f"${stats.get('avg_price', 0):.2f}")
    except Exception as e:
        st.sidebar.error(f"Stats Error: {e}")
    
    # 🎛️ Search Configuration
    st.sidebar.header("🎛️ Search Settings")
    limit = st.sidebar.slider("Results Limit", 1, 20, 10)
    show_embeddings = st.sidebar.checkbox("Show Embedding Analysis", False)
    
    # 📤 Add New Product Section
    st.header("📤 Add New Product")
    st.markdown("*Upload a new product to demonstrate real-time embedding generation and search capability*")
    
    with st.expander("🆕 Upload New Product", expanded=False):
        with st.form("add_product_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("📝 Product Title*", placeholder="e.g., Smart WiFi LED Light Bulb")
                brand = st.text_input("🏷️ Brand*", placeholder="e.g., Philips")
                price = st.number_input("💰 Price*", min_value=0.01, value=25.99, step=0.01)
                main_category = st.selectbox("📂 Main Category*", [
                    "Tools", "Appliances", "Lighting", "Plumbing", "Electrical", 
                    "Hardware", "Garden", "Clothing", "Electronics", "General"
                ])
            
            with col2:
                description = st.text_area("📄 Description*", 
                                         placeholder="Detailed product description...", 
                                         height=100)
                image_url = st.text_input("🖼️ Image URL", 
                                        placeholder="https://example.com/product-image.jpg")
                sub_category = st.text_input("📂 Sub Category", 
                                           placeholder="e.g., Smart Home, LED Bulbs")
            
            st.markdown("**Required fields marked with ***")
            submitted = st.form_submit_button("🚀 Add Product & Generate Embeddings", type="primary")
            
            if submitted:
                # Validate required fields
                if not all([title, brand, price, main_category, description]):
                    st.error("❌ Please fill in all required fields marked with *")
                else:
                    with st.spinner("🧠 Generating embeddings and adding to database..."):
                        product_data = {
                            'title': title,
                            'description': description,
                            'brand': brand,
                            'price': price,
                            'image_url': image_url or "https://via.placeholder.com/300x300?text=No+Image",
                            'main_category': main_category,
                            'sub_category': sub_category or f"{main_category} - Custom"
                        }
                        
                        result = search_engine.add_new_product(product_data)
                        
                        if result['success']:
                            st.success(f"✅ {result['message']}")
                            st.info(f"🧠 Generated {result['embedding_size']}-dimensional embedding")
                            st.info(f"🆔 Product ID: {result['product_id']}")
                            
                            # Test immediate search
                            st.markdown("---")
                            st.markdown("**🔍 Test Search for Your New Product:**")
                            if st.button(f"Search for '{title[:30]}...'"):
                                search_results = search_engine.semantic_search(title, limit=5)
                                if search_results:
                                    st.markdown("**Search Results:**")
                                    for i, product in enumerate(search_results['results'][:3]):
                                        if product['product_id'] == result['product_id']:
                                            st.success(f"✅ Found your new product! (Rank #{i+1}, Similarity: {product['similarity']:.4f})")
                                        else:
                                            st.info(f"#{i+1}: {product['title'][:50]}... (Similarity: {product['similarity']:.4f})")
                        else:
                            st.error(f"❌ Failed to add product: {result['error']}")
    
    st.markdown("---")
    
    # 🔍 Search Interface
    st.header("🔍 Semantic Product Search")
    
    query = st.text_input(
        "Search for products:",
        placeholder="e.g., 'power drill for home projects', 'energy efficient refrigerator', 'ceiling fan with lights'",
        help="Enter a natural language description of what you're looking for"
    )
    
    search_button = st.button("🚀 Search", type="primary")
    
    # Perform search
    if search_button and query:
        with st.spinner("🔍 Searching with HNSW vector index..."):
            search_results = search_engine.semantic_search(query, limit=limit)
            
            if search_results:
                display_search_results_with_embeddings(search_results, show_embeddings)
            else:
                st.error("❌ Search failed")
    
    # 📈 Performance History
    if search_engine.search_history:
        st.header("📈 Search Performance History")
        
        df = pd.DataFrame(search_engine.search_history)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(df, x='timestamp', y='total_time', 
                         title='Search Performance Over Time',
                         labels={'total_time': 'Total Time (s)', 'timestamp': 'Time'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df, x='timestamp', y='num_results',
                        title='Number of Results Over Time',
                        labels={'num_results': 'Results Count', 'timestamp': 'Time'})
            st.plotly_chart(fig, use_container_width=True)
    
    # 🎯 Example Searches
    st.header("🎯 Try These Example Searches")
    example_queries = [
        "power drill for home renovation",
        "energy efficient kitchen appliances", 
        "ceiling fan with LED lighting",
        "stainless steel refrigerator",
        "cordless tools for DIY projects"
    ]
    
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(f"'{example}'", key=f"example_{i}"):
                search_results = search_engine.semantic_search(example, limit=limit)
                if search_results:
                    display_search_results_with_embeddings(search_results, show_embeddings)

if __name__ == "__main__":
    main()
