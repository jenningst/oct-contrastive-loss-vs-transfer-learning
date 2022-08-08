import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath("./"))

from src.classifier.inference import api

client = TestClient(api)

def test_liveness():
    response = client.get("/")
    assert response.status_code == 200