"""
Example usage of the document retrieval functions.
"""

from retrieve_docs import retrieve_relevant_docs, retrieve_with_sql_filter


def main():
    # Example 1: Simple semantic search
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Semantic Search")
    print("=" * 70)

    query = "How to get a caste certificate?"
    print(f"\nQuery: {query}\n")

    results = retrieve_relevant_docs(query=query, top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Department: {doc['department']}")
        print(f"  Service: {doc['service_type']}")
        print(f"  Text: {doc['chunk_text'][:120]}...")
        print(f"  URL: {doc['source_url']}\n")

    # Example 2: Filtered search by department
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Search with Department Filter")
    print("=" * 70)

    query = "certificate services"
    department = "General Administration Department"
    print(f"\nQuery: {query}")
    print(f"Department Filter: {department}\n")

    results = retrieve_relevant_docs(
        query=query,
        department_filter=department,
        top_k=2,
    )
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Department: {doc['department']}")
        print(f"  Service: {doc['service_type']}")
        print(f"  Text: {doc['chunk_text'][:120]}...\n")

    # Example 3: Advanced SQL filtering
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Advanced SQL Filtering")
    print("=" * 70)

    query = "licence registration"
    department = "Labour Resources Department"
    service_type = "Other Services"
    print(f"\nQuery: {query}")
    print(f"Department: {department}")
    print(f"Service Type: {service_type}\n")

    results = retrieve_with_sql_filter(
        query=query,
        department=department,
        service_type=service_type,
        top_k=2,
    )
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Department: {doc['department']}")
        print(f"  Service: {doc['service_type']}")
        print(f"  Similarity Score: {doc['similarity_score']:.4f}")
        print(f"  Text: {doc['chunk_text'][:120]}...\n")


if __name__ == "__main__":
    main()
