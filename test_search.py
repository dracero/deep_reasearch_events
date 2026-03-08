import asyncio
from agents.crawl4ai_utils import get_crawler, close_crawler
from crawl4ai import CrawlerRunConfig, CacheMode
from urllib.parse import quote_plus

async def main():
    crawler = await get_crawler()
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    query = "site:promiedos.com.ar libertadores sudamericana 2024-02-28"
    
    print("Fetching Google...")
    url = f"https://www.google.com/search?q={quote_plus(query)}&gl=ar&hl=es&num=10"
    res = await crawler.arun(url=url, config=run_config)
    
    if res and res.markdown:
        md = res.markdown.raw_markdown if hasattr(res.markdown, 'raw_markdown') else str(res.markdown)
        with open("google_debug.md", "w") as f:
            f.write(md)
        print("Done writing google_debug.md")
    else:
        print("No content returned.")
        
    await close_crawler()

if __name__ == "__main__":
    asyncio.run(main())
