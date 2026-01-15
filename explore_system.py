#!/usr/bin/env python3
"""
Comprehensive DocQuery AI System Explorer
This script uploads sample documents and tests all functionality
"""
import requests
import json
import os
import time

def upload_document(base_url, file_path):
    """Upload a document to the API"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f'{base_url}/upload', files=files)
            return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def explore_system():
    print('🚀 === EXPLORING DOCQUERY AI SYSTEM === 🚀')
    print()
    
    base_url = 'http://localhost:8001'
    
    # 1. Health Check
    print('🏥 HEALTH CHECK')
    print('-' * 40)
    try:
        response = requests.get(f'{base_url}/health')
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        print('✅ API is healthy and running')
    except Exception as e:
        print(f'❌ Health check failed: {e}')
        return
    print()
    
    # 2. Check for Sample Documents (Optional)
    print('📤 CHECKING FOR SAMPLE DOCUMENTS')
    print('-' * 40)
    sample_docs_dir = 'sample_docs'
    
    if os.path.exists(sample_docs_dir):
        sample_files = [f for f in os.listdir(sample_docs_dir) if f.endswith(('.txt', '.pdf', '.docx'))]
        print(f'📁 Found {len(sample_files)} sample documents')
        
        uploaded_docs = []
        for filename in sample_files[:3]:  # Upload up to 3 samples
            doc_path = os.path.join(sample_docs_dir, filename)
            success, result = upload_document(base_url, doc_path)
            if success:
                print(f'✅ Uploaded: {filename}')
                uploaded_docs.append(filename)
            else:
                print(f'❌ Failed to upload {filename}: {result}')
        print(f'📁 Total uploaded: {len(uploaded_docs)} documents')
    else:
        print('📄 No sample documents found - system ready for your documents!')
        print('   💡 Tip: Upload your own documents through the web interface')
    print()
    
    # 3. List All Documents
    print('📋 DOCUMENT LIBRARY')
    print('-' * 40)
    try:
        response = requests.get(f'{base_url}/documents')
        docs = response.json()
        print(f'📚 Found {len(docs)} documents in library:')
        for i, doc in enumerate(docs):
            filename = doc.get('filename', 'Unknown')
            doc_id = doc.get('id', 'No ID')
            upload_time = doc.get('upload_time', 'Unknown')
            print(f'  [{i+1}] {filename}')
            print(f'      ID: {doc_id}')
            print(f'      Uploaded: {upload_time}')
            print()
    except Exception as e:
        print(f'❌ Failed to list documents: {e}')
        print()
    
    # 4. Test Search Functionality
    print('🔍 SEARCH FUNCTIONALITY TESTING')
    print('-' * 40)
    
    test_queries = [
        'python programming',
        'machine learning',
        'API development', 
        'supervised learning',
        'Flask framework',
        'HTTP methods',
        'neural networks',
        'REST API'
    ]
    
    for query in test_queries:
        try:
            response = requests.get(f'{base_url}/search', params={'query': query})
            results = response.json()
            print(f'Query: "{query}"')
            print(f'  Results: {len(results)} matches found')
            
            # Show top results
            for i, result in enumerate(results[:2]):  # Top 2 results
                filename = result.get('filename', 'Unknown')
                score = result.get('score', 0)
                snippet = result.get('content', '')[:100] + '...' if result.get('content') else 'No content'
                print(f'    [{i+1}] {filename} (Score: {score:.3f})')
                print(f'        "{snippet}"')
            print()
        except Exception as e:
            print(f'❌ Search failed for "{query}": {e}')
            print()
    
    # 5. System Statistics
    print('📊 SYSTEM STATISTICS')
    print('-' * 40)
    try:
        response = requests.get(f'{base_url}/stats')
        stats = response.json()
        print('System Performance:')
        for key, value in stats.items():
            print(f'  {key}: {value}')
    except Exception as e:
        print(f'❌ Failed to get statistics: {e}')
    print()
    
    # 6. API Endpoints Discovery
    print('🔧 AVAILABLE API ENDPOINTS')
    print('-' * 40)
    endpoints = [
        ('GET', '/health', 'Health check'),
        ('GET', '/docs', 'API documentation'),
        ('GET', '/documents', 'List all documents'),
        ('POST', '/upload', 'Upload document'),
        ('GET', '/search', 'Search documents'),
        ('GET', '/stats', 'System statistics'),
        ('DELETE', '/documents/{id}', 'Delete document')
    ]
    
    for method, path, description in endpoints:
        print(f'  {method:6} {path:20} - {description}')
    print()
    
    print('🎉 === EXPLORATION COMPLETE === 🎉')
    print()
    print('🌐 ACCESS INTERFACES:')
    print(f'   • API Documentation: http://localhost:8001/docs')
    print(f'   • Streamlit Web App: http://localhost:8502')
    print()
    print('💡 WHAT YOU CAN DO NEXT:')
    print('   • Open the web interface to interact visually')
    print('   • Try more complex search queries')
    print('   • Upload your own documents')
    print('   • Explore the interactive API docs')

if __name__ == '__main__':
    explore_system()