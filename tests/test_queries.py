# mtg_search/tests/test_queries.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_db.query_vector_db import query_vector_db

def test_queries():
    # List of test queries
    test_queries = [
        "red cards with flicker-like effects",
        "blue creatures with flying",
        "green cards that draw cards",
        "artifacts with tap abilities",
        "black removal spells"
    ]
    top_k = 5

    # Run each query and print results
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = query_vector_db(query, top_k)
        print("Top matches:")
        for result in results:
            print(f"- {result['name']} (Similarity: {result['similarity']:.4f})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run a single query from command line
        query = sys.argv[1]
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        results = query_vector_db(query, top_k)
        print(f"\nQuery: {query}")
        print("Top matches:")
        for result in results:
            print(f"- {result['name']} (Similarity: {result['similarity']:.4f})")
    else:
        # Run demo queries
        test_queries()