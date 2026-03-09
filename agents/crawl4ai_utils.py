"""
Shared crawl4ai utilities for search and scraping.
Replaces: Serper, Tavily, DuckDuckGo, Jina Reader, Firecrawl.

Strategy:
  - Search: Scrape Google/DuckDuckGo HTML search results pages
  - Scrape: Use crawl4ai's AsyncWebCrawler for JS-rendered pages with clean markdown
"""

import asyncio
import logging
import re
from typing import List, Optional
from urllib.parse import quote_plus, urljoin

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Singleton Crawler
# ─────────────────────────────────────────────────────

_crawler_instance: Optional[AsyncWebCrawler] = None
_crawler_lock = asyncio.Lock()


async def get_crawler() -> AsyncWebCrawler:
    """Get or create a singleton AsyncWebCrawler instance."""
    global _crawler_instance
    async with _crawler_lock:
        if _crawler_instance is None:
            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
            )
            _crawler_instance = AsyncWebCrawler(config=browser_config)
            await _crawler_instance.__aenter__()
            logger.info("🕷️ crawl4ai AsyncWebCrawler initialized")
        return _crawler_instance


async def close_crawler():
    """Close the singleton crawler (call on shutdown)."""
    global _crawler_instance
    async with _crawler_lock:
        if _crawler_instance is not None:
            try:
                await _crawler_instance.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing crawler: {e}")
            _crawler_instance = None
            logger.info("🕷️ crawl4ai AsyncWebCrawler closed")


async def _reset_crawler():
    """Reset the crawler instance after a browser crash/close."""
    global _crawler_instance
    async with _crawler_lock:
        if _crawler_instance is not None:
            try:
                await _crawler_instance.__aexit__(None, None, None)
            except Exception:
                pass
            _crawler_instance = None
            logger.info("🔄 Crawler instance reset after browser close")


# ─────────────────────────────────────────────────────
# Search: Google → DuckDuckGo HTML fallback
# ─────────────────────────────────────────────────────

# Phrases that indicate Google returned a "no results" page
_GOOGLE_NO_RESULTS_PHRASES = [
    "No se han encontrado resultados",
    "did not match any documents",
    "No results found",
    "no encontró ningún documento",
]


def _is_browser_closed_error(error: Exception) -> bool:
    """Check if an exception indicates the browser/page was closed."""
    msg = str(error).lower()
    return any(kw in msg for kw in ["closed", "connection closed", "target page"])


def _parse_google_results(markdown: str, max_results: int = 10) -> List[dict]:
    """Parse Google search results from raw markdown content."""
    from urllib.parse import unquote
    results = []
    
    # Match any link [Text](URL) with nested brackets allowance
    links = list(re.finditer(r'\[((?:[^\[\]]|\[[^\]]*\])*)\]\((https?://[^\)]+)\)', markdown.replace('\n', ' ')))
    for idx, m in enumerate(links):
        if len(results) >= max_results: break
        
        title = m.group(1).strip()
        url = m.group(2).strip()
        
        if len(title) > 10 and not any(skip in url for skip in ['google.', 'gstatic.', 'youtube.', 'accounts.']):
            # cleanup title
            title = re.sub(r'!\[.*?\]\([^\)]+\)', '', title).strip() # remove images
            if title and not "Ver más" in title and not "Siguiente" in title:
                # Get snippet from surrounding text
                start_idx = m.end()
                end_idx = links[idx+1].start() if idx + 1 < len(links) else len(markdown)
                snippet_raw = markdown[start_idx:end_idx].replace('\n', ' ')
                snippet_clean = re.sub(r'\[.*?\]\([^\)]+\)', '', snippet_raw).strip() # remove other links from snippet
                snippet_clean = re.sub(r'\s+', ' ', snippet_clean)[:300]
                
                # Check for duplicates
                if not any(r['url'] == url for r in results):
                    results.append({
                        "title": title[:200],
                        "url": url,
                        "content": snippet_clean,
                    })
    return results


def _parse_ddg_html_results(markdown: str, max_results: int = 10) -> List[dict]:
    """Parse DuckDuckGo HTML search results from markdown."""
    from urllib.parse import unquote
    results = []
    
    # Match ## [Title](url) allowing nested brackets in Title
    matches = list(re.finditer(r'##\s+\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^\)]+)\)', markdown))
    for idx, m in enumerate(matches):
        if len(results) >= max_results: break
        
        title = m.group(1).strip()
        url = m.group(2).strip()
        
        # extract uddg redirect
        uddg_match = re.search(r'uddg=([^&]+)', url)
        if uddg_match:
            actual_url = unquote(uddg_match.group(1))
            title = re.sub(r'!\[.*?\]\([^\)]+\)', '', title).strip() # remove images
            
            # extract snippet
            start_idx = m.end()
            end_idx = matches[idx+1].start() if idx + 1 < len(matches) else len(markdown)
            snippet_raw = markdown[start_idx:end_idx].replace('\n', ' ')
            snippet_clean = re.sub(r'\[.*?\]\([^\)]+\)', '', snippet_raw).strip() 
            snippet_clean = re.sub(r'\s+', ' ', snippet_clean)[:300]
            
            # Check for duplicates
            if not any(r['url'] == actual_url for r in results):
                results.append({
                    "title": title[:200],
                    "url": actual_url,
                    "content": snippet_clean,
                })
    return results


async def crawl4ai_search(
    query: str,
    max_results: int = 8,
    gl: str = "ar",
    hl: str = "es",
    include_domains: Optional[List[str]] = None,
) -> List[dict]:
    """Search the web using crawl4ai to scrape search engines."""
    import uuid
    crawler = await get_crawler()
    session_id = f"search_{uuid.uuid4().hex}"

    effective_query = query
    if include_domains:
        site_query = " OR ".join([f"site:{d}" for d in include_domains])
        effective_query = f"{query} ({site_query})"

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.3,
                threshold_type="fixed",
                min_word_threshold=0,
            )
        ),
    )

    # 1. Try Google
    google_url = (
        f"https://www.google.com/search?"
        f"q={quote_plus(effective_query)}&gl={gl}&hl={hl}&num={max_results + 5}"
    )

    try:
        result = await crawler.arun(url=google_url, config=run_config, session_id=session_id)
        if result and result.markdown:
            md = result.markdown.raw_markdown if hasattr(result.markdown, 'raw_markdown') else str(result.markdown)
            # Detect Google "no results" page
            if any(phrase in md for phrase in _GOOGLE_NO_RESULTS_PHRASES):
                logger.warning(f"⚠️ Google returned NO RESULTS page for '{query}' (query too restrictive?)")
            else:
                parsed = _parse_google_results(md, max_results)
                if parsed:
                    logger.info(f"🔍 Google search '{query}': {len(parsed)} results")
                    return parsed[:max_results]
                logger.warning(f"⚠️ Google returned content but no results parsed for '{query}'")
    except Exception as e:
        if _is_browser_closed_error(e):
            logger.warning(f"⚠️ Browser closed during Google search '{query}'. Resetting crawler.")
            await _reset_crawler()
            crawler = await get_crawler()  # get a fresh instance
        else:
            logger.warning(f"⚠️ Google search failed for '{query}': {e}")

    # 2. Fallback: DuckDuckGo HTML
    ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(effective_query)}"

    try:
        result = await crawler.arun(url=ddg_url, config=run_config, session_id=session_id)
        if result and result.markdown:
            md = result.markdown.raw_markdown if hasattr(result.markdown, 'raw_markdown') else str(result.markdown)
            parsed = _parse_ddg_html_results(md, max_results)
            if parsed:
                logger.info(f"🔍 DDG search '{query}': {len(parsed)} results")
                return parsed[:max_results]
            logger.info(f"ℹ️ DDG returned 0 results for '{query}'")
    except Exception as e:
        if _is_browser_closed_error(e):
            logger.warning(f"⚠️ Browser closed during DDG search '{query}'. Resetting crawler.")
            await _reset_crawler()
        else:
            logger.warning(f"⚠️ DDG search failed for '{query}': {e}")

    logger.warning(f"❌ All search engines failed for '{query}'")
    return []


# ─────────────────────────────────────────────────────
# Scrape: Single page → clean markdown
# ─────────────────────────────────────────────────────

async def crawl4ai_scrape(url: str, max_chars: int = 5000) -> str:
    """Scrape a single URL and return clean markdown content."""
    import uuid
    crawler = await get_crawler()
    session_id = f"scrape_{uuid.uuid4().hex}"

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        magic=True,
    )

    try:
        result = await crawler.arun(url=url, config=run_config, session_id=session_id)
        if result and result.markdown:
            md = result.markdown.fit_markdown if hasattr(result.markdown, 'fit_markdown') else str(result.markdown)
            if not md or len(md) < 50:
                md = result.markdown.raw_markdown if hasattr(result.markdown, 'raw_markdown') else str(result.markdown)
            if md and len(md) > 50:
                text = md[:max_chars]
                logger.info(f"✅ crawl4ai scraped {url} ({len(text)} chars)")
                return text
            logger.warning(f"⚠️ crawl4ai: too little content from {url}")
    except Exception as e:
        if _is_browser_closed_error(e):
            logger.warning(f"⚠️ Browser closed during scrape of {url}. Resetting crawler.")
            await _reset_crawler()
        else:
            logger.warning(f"❌ crawl4ai scrape failed for {url}: {e}")

    return ""


async def crawl4ai_scrape_batch(urls: List[str], max_chars: int = 5000) -> List[str]:
    """Scrape multiple URLs concurrently with crawl4ai."""
    tasks = [crawl4ai_scrape(url, max_chars) for url in urls]
    return await asyncio.gather(*tasks)
