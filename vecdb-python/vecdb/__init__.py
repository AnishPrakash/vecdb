"""VecDB Python Client SDK."""
import requests
from typing import List, Dict, Any, Optional

class VectorDB:
    """Minimal Python client for the vecdb REST API."""
    
    def __init__(self, host: str = 'http://localhost:8080'):
        self.host = host.rstrip('/')
        self.session = requests.Session()

    def insert(
        self,
        id: int,
        vector: List[float],
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Insert a vector with optional metadata payload."""
        resp = self.session.post(
            f'{self.host}/insert',
            json={'id': id, 'vector': vector, 'payload': payload or {}}
        )
        resp.raise_for_status()
        return resp.json()

    def search(
        self,
        vector: List[float],
        top_k: int = 10,
        ef: int = 50
    ) -> List[Dict]:
        """Search for top_k nearest neighbours."""
        resp = self.session.post(
            f'{self.host}/search',
            json={'vector': vector, 'top_k': top_k, 'ef': ef}
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        """Check if server is alive."""
        try:
            return self.session.get(f'{self.host}/health').ok
        except Exception:
            return False

# Usage example:
# db = VectorDB('http://localhost:8080')
# db.insert(id=1, vector=[0.1]*128, payload={'text': 'hello world'})
# results = db.search(vector=[0.1]*128, top_k=5)
# print(results)