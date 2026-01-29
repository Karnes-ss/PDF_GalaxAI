from api import create_app
from store import ScholarStore

store = ScholarStore()
app = create_app(store)
