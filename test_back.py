import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_upload_file():
    # Test case 1: Vector index is not initialized
    response = client.post("/qna/", json={"querie": "How are you?"})
    assert response.status_code == 200
    assert response.json() == {"error": "Vector index not initialized"}

    # Test case 2: Vector index is initialized
    # Mock the vector_index.get_relevant_documents() method
    # to return a list of documents
    def mock_get_relevant_documents(querie):
        return ["Document 1", "Document 2", "Document 3"]

    app.vector_index.get_relevant_documents = mock_get_relevant_documents

    response = client.post("/qna/", json={"querie": "How are you?"})
    assert response.status_code == 200
    assert response.json() == {"response": "How are you?"}

    # Additional test cases can be added here