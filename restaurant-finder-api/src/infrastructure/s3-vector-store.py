import boto3
from config import settings

class S3VectorStoreClientWrapper:
    def __init__(self):
        self.s3_client = boto3.client('s3vectors')
        self.bucket_name = settings.S3_VECTOR_STORE_BUCKET
        if not self.bucket_name:
            raise ValueError("S3_VECTOR_STORE_BUCKET environment variable is not set.")
        self.indexName = settings.S3_VECTOR_STORE_INDEX_NAME
        if not self.indexName:
            raise ValueError("S3_VECTOR_STORE_INDEX_NAME environment variable is not set.")
    

    def query(self, top_k: int, vector: list[float]) -> list[dict]:
        response = self.s3_client.query_vectors(
            bucketName=self.bucket_name,
            indexName=self.indexName,
            topK=top_k,
            queryVector={
                'float32': vector
            },
            returnMetadata=True
        )
        return response['Results']

    def insert(self, key: str, vector: list[float], metadata: dict) -> None:
        self.s3_client.put_vectors(
            vectorBucketName=self.bucket_name,
            indexName=self.indexName,
            vectors= [
                {
                    'id': key,
                    'vector': {'float32': vector},
                    'metadata': metadata
                }
            ]
        )

    def insertInBulk(self, items: list[dict]) -> None:
        vectors = []
        for item in items:
            vectors.append(
                {
                    'id': item['id'],
                    'vector': {'float32': item['vector']},
                    'metadata': item['metadata']
                }
            )
        self.s3_client.put_vectors(
            vectorBucketName=self.bucket_name,
            indexName=self.indexName,
            vectors=vectors
        )