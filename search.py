from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 5):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        return list(results)
