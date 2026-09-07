import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pdfplumber
from playwright.async_api import async_playwright, Browser, TimeoutError as PlaywrightTimeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DataIngestion")

RAW_DATA_DIR = Path("data/raw_scrapes")
PDF_DATA_DIR = Path("data/pdfs")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_DATA_DIR.mkdir(parents=True, exist_ok=True)


class IngestionEngine:
    def __init__(self, headless: bool = True, timeout_ms: int = 45000):
        self.headless = headless
        self.timeout_ms = timeout_ms

    async def scrape_web_page(
        self,
        url: str,
        scheme_name: str,
        level: str,
        category: str,
        target_selectors: Optional[List[str]] = None
    ) -> Optional[Dict]:
        logger.info(f"Initiating scrape for scheme: {scheme_name} at URL: {url}")
        target_selectors = target_selectors or ["article", "main", "#content", ".entry-content", "body"]
        
        async with async_playwright() as p:
            browser: Browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0")
            page = await context.new_page()
            
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                await page.wait_for_timeout(2000)

                extracted_text = ""
                for selector in target_selectors:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        extracted_text = "\n\n".join([await el.inner_text() for el in elements if await el.inner_text()])
                        if extracted_text.strip():
                            break

                if not extracted_text.strip():
                    extracted_text = await page.inner_text("body")

                clean_text = " ".join(extracted_text.split())
                
                payload = {
                    "text": clean_text,
                    "metadata": {
                        "source": url,
                        "scheme_name": scheme_name,
                        "level": level,
                        "category": category,
                        "type": "web"
                    }
                }
                
                output_file = RAW_DATA_DIR / f"{scheme_name.lower().replace(' ', '_')}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"Successfully scraped & cached: {output_file}")
                return payload

            except PlaywrightTimeout:
                logger.error(f"Timeout while attempting to fetch: {url}")
                return None
            except Exception as e:
                logger.critical(f"Unhandled error scraping {url}: {str(e)}", exc_info=True)
                return None
            finally:
                await context.close()
                await browser.close()

    @staticmethod
    def extract_pdf(
        file_path: str,
        scheme_name: str,
        level: str,
        category: str,
        source_url: Optional[str] = None
    ) -> List[Dict]:
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            logger.error(f"Target PDF file does not exist: {file_path}")
            return []

        extracted_pages = []
        logger.info(f"Extracting PDF: {pdf_path.name} for scheme: {scheme_name}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    raw_text = page.extract_text()
                    if not raw_text or not raw_text.strip():
                        continue
                        
                    clean_text = " ".join(raw_text.split())
                    extracted_pages.append({
                        "text": clean_text,
                        "metadata": {
                            "source": source_url or pdf_path.name,
                            "page": idx,
                            "scheme_name": scheme_name,
                            "level": level,
                            "category": category,
                            "type": "pdf"
                        }
                    })
            logger.info(f"Successfully extracted {len(extracted_pages)} pages from {pdf_path.name}")
            return extracted_pages
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {str(e)}", exc_info=True)
            return []


if __name__ == "__main__":
    engine = IngestionEngine(headless=True)
    asyncio.run(
        engine.scrape_web_page(
            url="https://pmkisan.gov.in/",
            scheme_name="PM-KISAN",
            level="Central",
            category="income_support"
        )
    )