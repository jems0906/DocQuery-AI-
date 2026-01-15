#!/usr/bin/env python3
"""
Test script to explore DocQuery AI API functionality
"""
import requests
import json
import time

def test_api():
    print('=== EXPLORING DOCQUERY AI API ===')
    print()
    
    base_url = 'http://localhost:8001'
    
    # Test health endpoint
    try:
        print('🏥 Testing Health Check...')
        response = requests.get(f'{base_url}/health')
        print(f'   Status: {response.status_code}')
        print(f'   Response: {response.json()}')
        print()
    except Exception as e:
        print(f'   ❌ Failed: {e}')
        print()
    
    # Test documents list endpoint
    try:
        print('📄 Testing Documents List...')
        response = requests.get(f'{base_url}/documents')
        print(f'   Status: {response.status_code}')
        docs = response.json()
        print(f'   Found {len(docs)} documents in database')
        for i, doc in enumerate(docs[:5]):  # Show first 5
            filename = doc.get('filename', 'Unknown')
            doc_id = doc.get('id', 'No ID')
            print(f'   [{i+1}] {filename} (ID: {doc_id})')
        if len(docs) > 5:
            print(f'   ... and {len(docs)-5} more documents')
        print()
    except Exception as e:
        print(f'   ❌ Failed: {e}')
        print()
    
    # Test search endpoint with different queries
    search_queries = ['python', 'machine learning', 'api', 'document', 'search']
    
    for query in search_queries:
        try:
            print(f'🔍 Testing Search: "{query}"')
            response = requests.get(f'{base_url}/search', params={'query': query})
            print(f'   Status: {response.status_code}')
            results = response.json()
            print(f'   Found {len(results)} results')
            
            # Show top 3 results
            for i, result in enumerate(results[:3]):
                filename = result.get('filename', 'Unknown')
                score = result.get('score', 0)
                print(f'   [{i+1}] {filename} (Score: {score:.3f})')
            print()
        except Exception as e:
            print(f'   ❌ Search failed for "{query}": {e}')
            print()
    
    # Test statistics endpoint
    try:
        print('📊 Testing Statistics...')
        response = requests.get(f'{base_url}/stats')
        print(f'   Status: {response.status_code}')
        stats = response.json()
        print(f'   Statistics: {json.dumps(stats, indent=2)}')
        print()
    except Exception as e:
        print(f'   ❌ Failed: {e}')
        print()
    
    print('=== API EXPLORATION COMPLETE ===')

if __name__ == '__main__':
    test_api()