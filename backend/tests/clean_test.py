from backend.scraper import scrape_urls_sync
from extractor import extract_all_sync
from models import CompanyRecord
from cleaner import clean 
from enricher import enrich_all_sync

url = "https://www.kaggle.com/madisrinija"

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


print("\n Cleaning \n\n")

clean_result = clean(company_record)
cleaned_record = clean_result.record

print(f"Cleaned Company Name: {cleaned_record.company_name}")
print(f"Cleaned Website: {cleaned_record.website}")
print(f"Cleaned Description: {cleaned_record.description}")
print(f"Cleaned Country: {cleaned_record.country}")
print(f"Cleaned Industry: {cleaned_record.industry}")
print(f"Cleaned Completeness: {cleaned_record.completeness_pct()}%")
print(f"Cleaned Missing Fields: {cleaned_record.missing_fields() or 'none'}")

print("\n \n enricher test \n\n")
enriched_record = enrich_all_sync(
        [(clean_result, scrape_result)]
)[0]
print(f"Enriched Company Name: {enriched_record.company_name}")
print(f"Enriched Website: {enriched_record.website}")
print(f"Enriched Description: {enriched_record.description}")
print(f"Enriched Country: {enriched_record.country}")
print(f"Enriched Industry: {enriched_record.industry}")
print(f"Enriched Completeness: {enriched_record.completeness}%")
print(f"Enriched scrape method: {enriched_record.scrape_method}")
print(f"Enriched ai_enriched: {enriched_record.ai_enriched}")
print(f"Enriched employee size: {enriched_record.employee_size}")

