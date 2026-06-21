from backend.scrapper import scrape_urls_sync
import sys

test_urls = sys.argv[1:] or [
        "https://openai.com",
        "https://github.com",
        "https://huggingface.co",
    ]
test_url = ["https://claude.ai/"]
results = scrape_urls_sync(test_urls)

for r in results:
    print(r)
    if r.success:
            print(f"  Title   : {r.title or '—'}")
            print(f"  Meta    : {(r.meta_description or '—')[:100]}")
            print(f"  Preview : {r.markdown[:400]} …")
    else:
            print(f"  Error   : {r.error}")
    print()

passed = sum(1 for r in results if r.success)
print(f"  Result: {passed}/{len(results)} succeeded")
