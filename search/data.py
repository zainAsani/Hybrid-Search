"""Load the product catalogue and the evaluation queries from the JSON files in data/."""
import json
from pathlib import Path

# Path(__file__) is this file. .parent.parent goes up two folders to the project root.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_products(path=DATA_DIR / "products.json"):
    """Return a list of products, e.g. [{"id": "p01", "title": "...", "description": "..."}, ...]."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_queries(path=DATA_DIR / "queries.json"):
    """Return the test queries, each with the ids of the products that are correct answers."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def product_text(product):
    """The text we actually search in: title and description joined together."""
    return f"{product['title']}. {product['description']}"
