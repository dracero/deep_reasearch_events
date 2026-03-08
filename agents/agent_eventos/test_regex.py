import re
from urllib.parse import unquote

def parse_ddg(md):
    results = []
    # Match ## [Title](url) allowing nested brackets in Title
    matches = re.finditer(r'##\s+\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^\)]+)\)', md)
    for m in matches:
        title = m.group(1).strip()
        url = m.group(2).strip()
        # extract uddg
        uddg_match = re.search(r'uddg=([^&]+)', url)
        if uddg_match:
            actual_url = unquote(uddg_match.group(1))
            title = re.sub(r'!\[.*?\]\([^\)]+\)', '', title).strip() # remove images
            results.append({"title": title[:200], "url": actual_url})
    return results

def parse_google(md):
    results = []
    # Match any link [Text](URL) with nested brackets allowance
    # \[( (?: [^\[\]] | \[[^\]]*\] )* )\]\((https?://[^\)]+)\)
    links = re.finditer(r'\[((?:[^\[\]]|\[[^\]]*\])*)\]\((https?://[^\)]+)\)', md.replace('\n', ' '))
    for m in links:
        title = m.group(1).strip()
        url = m.group(2).strip()
        if len(title) > 10 and not any(skip in url for skip in ['google.', 'gstatic.', 'youtube.']):
            # cleanup title
            title = re.sub(r'!\[.*?\]\([^\)]+\)', '', title).strip() # remove images
            if title and not "Ver más" in title and not "Siguiente" in title:
                results.append({"title": title[:200], "url": url})
            
    # Deduplicate by url
    seen = set()
    dedup = []
    for r in results:
        if r['url'] not in seen:
            seen.add(r['url'])
            dedup.append(r)
    return dedup

with open('ddg_out.md', 'r') as f:
    ddg = f.read()

with open('google_out.md', 'r') as f:
    google = f.read()

print("DDG Results: ", len(parse_ddg(ddg)))
for r in parse_ddg(ddg):
    print(r)

print("\nGoogle Results: ", len(parse_google(google)))
for r in parse_google(google):
    print(r)

