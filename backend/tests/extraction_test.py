from backend.scraper import scrape_urls_sync
from extractor import extract_all_sync

url = "https://www.whatsapp.com"

scrape_results = scrape_urls_sync([url])
extracted_results = extract_all_sync(scrape_results)

for scrape_result, company_record in extracted_results:
    print(f"URL: {scrape_result.url}")
    if company_record:
        print(f"Company Name: {company_record.company_name}")
        print(f"Website: {company_record.website}")
        print(f"Description: {company_record.description}")
        print(f"Country: {company_record.country}")
        print(f"Industry: {company_record.industry}")
        print(f"Completeness: {company_record.completeness_pct()}%")
        print(f"Missing Fields: {company_record.missing_fields() or 'none'}")