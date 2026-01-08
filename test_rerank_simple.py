#!/usr/bin/env python3
import json
import boto3

client = boto3.client('bedrock-runtime', region_name='us-east-1')

# Request body con api_version=2 (required)
request = {
    "query": "What is RAG?",
    "documents": [
        "RAG combines retrieval and generation",
        "CNNs are used in computer vision"
    ],
    "top_n": 2,
    "api_version": 2
}

print("Invocando Cohere Rerank v3.5...")
print(f"Request: {json.dumps(request, indent=2)}\n")

try:
    response = client.invoke_model(
        modelId='cohere.rerank-v3-5:0',
        body=json.dumps(request)
    )

    result = json.loads(response['body'].read())
    print("✅ ÉXITO!")
    print(json.dumps(result, indent=2))

except Exception as e:
    print(f"❌ Error: {e}")
