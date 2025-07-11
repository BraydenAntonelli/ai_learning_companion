from retriever.semantic_search import search_memory

print("Ask me a question (type 'exit' to quit):\n")

while True:
    query = input("→ Question: ")
    if query.strip().lower() == "exit":
        break

    results = search_memory(query, top_k=3)

    print("Top matches:")
    for match, dist in results:
        print(f"→ {match} (distance: {round(dist, 4)})")
    print()
