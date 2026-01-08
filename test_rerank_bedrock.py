#!/usr/bin/env python3
"""
Test de Cohere Rerank 3.5 Serverless en AWS Bedrock
Pricing: $2.00 por 1,000 queries (hasta 100 chunks por query)
"""

import json
import sys

import boto3

# Cliente Bedrock Runtime
client = boto3.client('bedrock-runtime', region_name='us-east-1')

# Query y documentos
query = "What is RAG and how does it improve LLM responses?"

documents = [
    "RAG (Retrieval Augmented Generation) combines retrieval with generation to enhance LLM accuracy",
    "Convolutional Neural Networks are used primarily in computer vision tasks",
    "Retrieval Augmented Generation retrieves relevant documents before generating responses",
    "Deep learning models require extensive training data and computational resources",
    "RAG systems use vector databases to store and retrieve semantic information"
]

# Request body - Cohere Rerank requiere api_version
request_body = {
    "query": query,
    "documents": documents,
    "top_n": 3,
    "return_documents": False
}

print("=" * 80)
print("🧪 TEST: Cohere Rerank 3.5 Serverless en Bedrock")
print("=" * 80)
print(f"\n📝 Query: {query}\n")
print(f"📚 Documentos ({len(documents)}):")
for i, doc in enumerate(documents):
    print(f"  [{i}] {doc[:70]}...")

print("\n🔄 Invocando modelo cohere.rerank-v3-5:0...")

try:
    response = client.invoke_model(
        modelId='cohere.rerank-v3-5:0',
        body=json.dumps(request_body)
    )

    # Parsear respuesta
    result = json.loads(response['body'].read())

    print("\n✅ Éxito!\n")
    print("📊 Resultados de Reranking:")
    print("-" * 80)

    if 'results' in result:
        for i, item in enumerate(result['results']):
            idx = item.get('index', 'N/A')
            score = item.get('relevance_score', 0)

            print(f"\n🏆 Rank #{i+1}")
            print(f"   Documento #{idx}")
            print(f"   Relevance Score: {score:.4f}")
            if idx != 'N/A' and idx < len(documents):
                print(f"   Texto: {documents[idx]}")

    print("\n" + "=" * 80)
    print("💰 Costo estimado:")
    print(f"   Queries: 1")
    print(f"   Documentos: {len(documents)} (< 100, cuenta como 1 query)")
    print(f"   Costo: $0.002 (1/1000 × $2.00)")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}")
    print(f"Mensaje: {str(e)}")
    print("\n💡 Si ves AccessDeniedException, intenta:")
    print("   1. Esperar 2-3 minutos (activación automática)")
    print("   2. Invocar el modelo de nuevo")
    sys.exit(1)
