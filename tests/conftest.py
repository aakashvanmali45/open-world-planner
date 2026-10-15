import pytest
from perception.embedder import Embedder

@pytest.fixture(scope = "session")
def clip_embedder():
    return Embedder()