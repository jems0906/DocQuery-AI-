import streamlit as st
import requests
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page config
st.set_page_config(
    page_title="Document Q&A Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .search-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .chunk-content {
        background-color: #ffffff;
        padding: 0.8rem;
        border-radius: 0.3rem;
        border: 1px solid #dee2e6;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def make_api_request(endpoint: str, method: str = "GET", data: dict = None, files: dict = None) -> dict:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, params=data)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return {}
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return {}


def check_api_health() -> bool:
    """Check if API is healthy"""
    try:
        response = make_api_request("/health")
        return response.get("status") == "healthy"
    except:
        return False


def main():
    """Main application"""
    st.markdown('<h1 class="main-header">📚 Document Q&A Engine</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API service is not available. Please start the FastAPI server first.")
        st.code("python -m uvicorn app.main:app --reload", language="bash")
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🔍 Search Documents", "📤 Upload Documents", "📊 Document Library", "📈 Analytics"]
    )
    
    if page == "🔍 Search Documents":
        search_page()
    elif page == "📤 Upload Documents":
        upload_page()
    elif page == "📊 Document Library":
        library_page()
    elif page == "📈 Analytics":
        analytics_page()


def search_page():
    """Search documents page"""
    st.header("🔍 Search Your Documents")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What is the main topic discussed in the document?",
            help="Enter your question and get answers from your uploaded documents"
        )
    
    with col2:
        max_results = st.selectbox("Max Results", [3, 5, 10], index=1)
    
    # Search options
    with st.expander("🔧 Advanced Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Get available documents
            documents = make_api_request("/documents")
            doc_options = {"All Documents": None}
            if documents:
                doc_options.update({
                    f"{doc['filename']} (ID: {doc['id']})": doc['id'] 
                    for doc in documents
                })
            
            selected_doc = st.selectbox("Search in specific document:", list(doc_options.keys()))
            document_id = doc_options[selected_doc]
        
        with col2:
            include_metadata = st.checkbox("Include metadata in results")
    
    # Search button and results
    if st.button("🔍 Search", type="primary") or query:
        if query.strip():
            with st.spinner("Searching documents..."):
                search_start = time.time()
                
                search_data = {
                    "query": query.strip(),
                    "max_results": max_results,
                    "include_metadata": include_metadata
                }
                
                if document_id:
                    search_data["document_id"] = document_id
                
                results = make_api_request("/search", "POST", search_data)
                
                search_time = time.time() - search_start
            
            if results:
                display_search_results(results, search_time)
            else:
                st.warning("No results found. Try a different query or upload more documents.")
        else:
            st.warning("Please enter a search query.")
    
    # Query suggestions
    if query and len(query) > 2:
        suggestions_data = make_api_request("/search/suggestions", "GET", {"q": query, "limit": 5})
        suggestions = suggestions_data.get("suggestions", [])
        
        if suggestions:
            st.subheader("💡 Query Suggestions")
            for suggestion in suggestions:
                if st.button(f"💭 {suggestion}", key=f"suggestion_{suggestion}"):
                    st.rerun()


def display_search_results(results: dict, search_time: float):
    """Display search results"""
    st.success(f"✅ Found {len(results.get('relevant_chunks', []))} relevant passages in {search_time:.2f}s")
    
    # Display answer
    if results.get("answer"):
        st.subheader("🎯 Answer")
        st.markdown(f'<div class="search-result">{results["answer"]}</div>', unsafe_allow_html=True)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Search Time", f"{results.get('search_time_ms', 0):.1f}ms")
    
    with col2:
        st.metric("Total Time", f"{results.get('total_time_ms', 0):.1f}ms")
    
    with col3:
        st.metric("Documents", results.get('document_count', 0))
    
    with col4:
        st.metric("Passages", len(results.get('relevant_chunks', [])))
    
    # Display relevant chunks
    st.subheader("📄 Relevant Passages")
    
    chunks = results.get('relevant_chunks', [])
    for i, chunk in enumerate(chunks):
        with st.expander(f"📋 Passage {i+1} (Score: {chunk.get('relevance_score', 0):.3f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f'<div class="chunk-content">{chunk["content"]}</div>', unsafe_allow_html=True)
            
            with col2:
                st.write("**Metadata:**")
                st.write(f"📄 Document ID: {chunk['document_id']}")
                st.write(f"📃 Page: {chunk.get('page_number', 'N/A')}")
                st.write(f"🔢 Tokens: {chunk.get('token_count', 0)}")
                st.write(f"🎯 Score: {chunk.get('relevance_score', 0):.3f}")
    
    # Feedback section
    st.subheader("👍 Was this helpful?")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("👍 Yes"):
            st.success("Thanks for your feedback!")
    
    with col2:
        if st.button("👎 No"):
            st.info("Thanks! We'll work on improving our results.")


def upload_page():
    """Document upload page"""
    st.header("📤 Upload Documents")
    
    st.markdown("""
    Upload your documents to make them searchable. Supported formats:
    - **PDF** - Text will be extracted automatically
    - **Text Files** - Plain text documents
    - **Word Documents** - DOCX format
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload one or more documents to add to your searchable library"
    )
    
    if uploaded_files:
        st.subheader(f"📁 Selected Files ({len(uploaded_files)})")
        
        # Display selected files
        for file in uploaded_files:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                st.write(f"{file.size / 1024:.1f} KB")
            with col3:
                st.write(file.type or "Unknown")
        
        # Upload button
        if st.button("🚀 Upload Documents", type="primary"):
            upload_documents(uploaded_files)


def upload_documents(files):
    """Upload documents with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    uploaded_docs = []
    failed_uploads = []
    
    for i, file in enumerate(files):
        try:
            status_text.text(f"Uploading {file.name}...")
            
            # Prepare file for upload
            files_data = {"file": (file.name, file.getvalue(), file.type)}
            
            # Upload document
            result = make_api_request("/documents", "POST", files=files_data)
            
            if result:
                uploaded_docs.append(result)
                st.success(f"✅ Successfully uploaded: {file.name}")
            else:
                failed_uploads.append(file.name)
                st.error(f"❌ Failed to upload: {file.name}")
            
        except Exception as e:
            failed_uploads.append(file.name)
            st.error(f"❌ Error uploading {file.name}: {e}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(files))
    
    status_text.text("Upload completed!")
    
    # Summary
    st.subheader("📊 Upload Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("✅ Successful", len(uploaded_docs))
    
    with col2:
        st.metric("❌ Failed", len(failed_uploads))
    
    # Display uploaded documents
    if uploaded_docs:
        st.subheader("📚 Newly Added Documents")
        for doc in uploaded_docs:
            with st.expander(f"📄 {doc['filename']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**File Type:** {doc['file_type']}")
                    st.write(f"**Size:** {doc['file_size'] / 1024:.1f} KB")
                    st.write(f"**Chunks:** {doc['total_chunks']}")
                
                with col2:
                    st.write(f"**Processed:** {'✅' if doc['processed'] else '❌'}")
                    st.write(f"**Processing Time:** {doc.get('processing_time', 0):.2f}s")
                    st.write(f"**Upload Time:** {doc['upload_time']}")


def library_page():
    """Document library page"""
    st.header("📊 Document Library")
    
    # Get documents
    documents = make_api_request("/documents")
    
    if not documents:
        st.info("📭 No documents uploaded yet. Go to the Upload page to add documents.")
        return
    
    # Library statistics
    st.subheader("📈 Library Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_docs = len(documents)
    total_size = sum(doc.get('file_size', 0) for doc in documents)
    total_chunks = sum(doc.get('total_chunks', 0) for doc in documents)
    avg_processing_time = sum(doc.get('processing_time', 0) for doc in documents) / max(total_docs, 1)
    
    with col1:
        st.metric("📄 Total Documents", total_docs)
    
    with col2:
        st.metric("💾 Total Size", f"{total_size / (1024 * 1024):.1f} MB")
    
    with col3:
        st.metric("🧩 Total Chunks", total_chunks)
    
    with col4:
        st.metric("⏱️ Avg Process Time", f"{avg_processing_time:.2f}s")
    
    # File type distribution
    st.subheader("📊 File Type Distribution")
    
    df = pd.DataFrame(documents)
    if not df.empty:
        file_type_counts = df['file_type'].value_counts()
        
        fig = px.pie(
            values=file_type_counts.values,
            names=file_type_counts.index,
            title="Document Types"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Documents table
    st.subheader("📋 Document List")
    
    # Search and filter
    search_term = st.text_input("🔍 Search documents:", placeholder="Filter by filename...")
    
    # Filter documents
    filtered_docs = documents
    if search_term:
        filtered_docs = [
            doc for doc in documents 
            if search_term.lower() in doc['filename'].lower()
        ]
    
    # Display documents
    for doc in filtered_docs:
        with st.expander(f"📄 {doc['filename']} ({doc['file_type'].upper()})"):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**Original Name:** {doc['original_filename']}")
                st.write(f"**File Size:** {doc['file_size'] / 1024:.1f} KB")
                st.write(f"**Upload Time:** {doc['upload_time']}")
            
            with col2:
                st.write(f"**Processed:** {'✅' if doc['processed'] else '❌'}")
                st.write(f"**Total Chunks:** {doc['total_chunks']}")
                st.write(f"**Processing Time:** {doc.get('processing_time', 0):.2f}s")
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{doc['id']}"):
                    if st.confirm(f"Delete {doc['filename']}?"):
                        delete_result = make_api_request(f"/documents/{doc['id']}", "DELETE")
                        if delete_result:
                            st.success("Document deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")
                
                if st.button(f"🔍 View Chunks", key=f"chunks_{doc['id']}"):
                    view_document_chunks(doc['id'], doc['filename'])


def view_document_chunks(document_id: int, filename: str):
    """View document chunks"""
    st.subheader(f"📋 Chunks for {filename}")
    
    chunks_data = make_api_request(f"/documents/{document_id}/chunks")
    chunks = chunks_data.get("chunks", [])
    
    if not chunks:
        st.info("No chunks found for this document.")
        return
    
    st.write(f"Found {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        with st.expander(f"Chunk {i+1} (Page: {chunk.get('page_number', 'N/A')})"):
            st.markdown(f'<div class="chunk-content">{chunk["content"][:500]}...</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Token Count:** {chunk['token_count']}")
                st.write(f"**Page Number:** {chunk.get('page_number', 'N/A')}")
            
            with col2:
                st.write(f"**Chunk Index:** {chunk['chunk_index']}")
                if chunk.get('metadata'):
                    st.write(f"**Metadata:** {chunk['metadata']}")


def analytics_page():
    """Analytics and insights page"""
    st.header("📈 Analytics & Insights")
    
    # Get analytics data
    stats_data = make_api_request("/analytics/stats")
    popular_queries = make_api_request("/analytics/popular-queries")
    
    if not stats_data:
        st.error("Failed to load analytics data.")
        return
    
    # Search engine statistics
    st.subheader("🔍 Search Engine Performance")
    
    search_stats = stats_data.get("search_engine_stats", {}).get("search_stats", {})
    vector_stats = stats_data.get("search_engine_stats", {}).get("vector_store_stats", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Searches", search_stats.get("total_searches", 0))
    
    with col2:
        st.metric("Avg Search Time", f"{search_stats.get('avg_search_time', 0):.1f}ms")
    
    with col3:
        st.metric("Total Vectors", vector_stats.get("total_vectors", 0))
    
    with col4:
        st.metric("Total Documents", vector_stats.get("total_documents", 0))
    
    # System information
    st.subheader("⚙️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 Embedding Model**")
        st.code(stats_data.get("search_engine_stats", {}).get("embedding_model", "Unknown"))
        
        st.markdown("**📏 Embedding Dimension**")
        st.code(str(stats_data.get("search_engine_stats", {}).get("embedding_dimension", "Unknown")))
    
    with col2:
        st.markdown("**💾 Vector Store Type**")
        st.code(vector_stats.get("index_type", "Unknown"))
        
        st.markdown("**🧩 Total Chunks**")
        st.code(str(vector_stats.get("total_chunks", "Unknown")))
    
    # Popular queries
    st.subheader("🔥 Popular Queries")
    
    queries = popular_queries.get("popular_queries", [])
    if queries:
        df_queries = pd.DataFrame(queries)
        
        fig = px.bar(
            df_queries,
            x="count",
            y="query",
            orientation="h",
            title="Most Popular Search Queries"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Query table
        st.markdown("**📊 Query Statistics**")
        st.dataframe(df_queries, use_container_width=True)
    else:
        st.info("No query data available yet. Start searching to see analytics!")
    
    # Cache management
    st.subheader("🗄️ Cache Management")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🗑️ Clear Cache"):
            result = make_api_request("/admin/clear-cache", "POST")
            if result:
                st.success("Cache cleared successfully!")
            else:
                st.error("Failed to clear cache")
    
    with col2:
        st.info("Clear the application cache to free up memory and ensure fresh results.")


if __name__ == "__main__":
    main()