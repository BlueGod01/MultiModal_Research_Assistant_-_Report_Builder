#Dependency:
from tavily import TavilyClient
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv()  # Load Tavily Environment variable from .env file

# -----------------------------
# Tavily Client
# -----------------------------

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# -----------------------------
# Tools
# -----------------------------
@tool
def tavily_search(query: str) -> str:
    """
    Uses TavilySearchResults if installed and TAVILY_API_KEY is set.
    Returns list of dict with common fields. Note: published date is often missing.
    """
    results = tavily_client.search(query=query, max_results=5)
    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


@tool
def tavily_extract(url: str, include_images: bool = True) -> List[Dict]:
    """Extract content from a given URL using Tavily."""
    response = tavily_client.extract(urls=[url],include_images=include_images)
    failed_urls = response.get("failed_results", [])
    if failed_urls:
        return (f"Failed to extract {len(failed_urls)} URLs.")
    extracted_data = []
    for result in response.get("results", []):
        data = {
            "source_url": result["url"],
            "content": result["raw_content"],
            "images": result.get("images", [])#List of image URLS.
        }
        extracted_data.append(data)
    return extracted_data