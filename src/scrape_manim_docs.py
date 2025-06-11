import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os
import re
from typing import Set, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ManimDocsCrawler:
    def __init__(self, 
                 start_url: str = "https://docs.manim.community/en/stable/",
                 output_dir: str = "documents",
                 delay: float = 0.5,
                 max_pages: Optional[int] = None):
        """
        Initialize the Manim documentation crawler.
        
        Args:
            start_url: Starting URL for crawling
            output_dir: Directory to save scraped content
            delay: Delay between requests (in seconds)
            max_pages: Maximum pages to scrape (None for unlimited)
        """
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.output_dir = output_dir
        self.delay = delay
        self.max_pages = max_pages
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created directory: {output_dir}")
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def clean_filename(self, url: str) -> str:
        """Create a clean filename from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if not path or path == 'index.html':
            return "index.txt"
        
        # Remove file extensions and replace special characters
        filename = re.sub(r'\.html?$', '', path)
        filename = re.sub(r'[^\w\-_./]', '_', filename)
        filename = filename.replace('/', '_')
        
        return f"{filename}.txt"
    
    def extract_content(self, url: str) -> Optional[str]:
        """Extract main content from a single page."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try multiple selectors for main content
            content_selectors = [
                'main#main-content',
                'div.main-content',
                'article',
                'div.content',
                'div.document',
                'div.body'
            ]
            
            content = None
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element
                    break
            
            # Fallback to body if no main content found
            if not content:
                content = soup.find('body')
            
            if content:
                # Remove navigation, footer, and other non-content elements
                for tag in content.find_all(['nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                
                # Remove script and style tags
                for tag in content.find_all(['script', 'style']):
                    tag.decompose()
                
                # Extract text
                text = content.get_text(separator='\n', strip=True)
                
                # Clean up excessive whitespace
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r' +', ' ', text)
                
                return text.strip()
            
            logger.warning(f"Could not find main content on {url}")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            return None
    
    def find_links(self, url: str) -> Set[str]:
        """Find all internal links on a page."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = set()
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Create absolute URL
                absolute_url = urljoin(url, href)
                
                # Remove fragments and query parameters for deduplication
                clean_url = absolute_url.split('#')[0].split('?')[0]
                
                # Check if it's an internal link
                if urlparse(clean_url).netloc == self.domain:
                    # Filter out non-documentation links
                    path = urlparse(clean_url).path
                    if (not path.endswith(('.pdf', '.zip', '.tar.gz')) and
                        not 'github.com' in clean_url and
                        not '_static' in path and
                        not '_downloads' in path):
                        links.add(clean_url)
            
            return links
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error finding links on {url}: {e}")
            return set()
    
    def save_content(self, url: str, content: str) -> bool:
        """Save content to file."""
        try:
            filename = self.clean_filename(url)
            filepath = os.path.join(self.output_dir, filename)
            
            # Add URL as header for reference
            full_content = f"Source: {url}\n\n{content}"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            logger.info(f"Saved: {filename} ({len(content)} chars)")
            return True
            
        except OSError as e:
            logger.error(f"Error saving file for {url}: {e}")
            return False
    
    def crawl(self) -> int:
        """
        Crawl the entire documentation website.
        
        Returns:
            Number of pages successfully scraped
        """
        urls_to_visit = [self.start_url]
        visited_urls = set()
        scraped_count = 0
        
        logger.info(f"Starting crawl from: {self.start_url}")
        logger.info(f"Max pages: {self.max_pages or 'unlimited'}")
        
        while urls_to_visit and (self.max_pages is None or scraped_count < self.max_pages):
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            
            logger.info(f"Crawling ({scraped_count + 1}): {current_url}")
            
            # Extract and save content
            content = self.extract_content(current_url)
            if content and len(content) > 100:  # Only save if substantial content
                if self.save_content(current_url, content):
                    scraped_count += 1
            
            # Find new links to crawl
            new_links = self.find_links(current_url)
            for link in new_links:
                if link not in visited_urls and link not in urls_to_visit:
                    urls_to_visit.append(link)
            
            # Rate limiting
            time.sleep(self.delay)
            
            # Progress update
            if scraped_count % 10 == 0:
                logger.info(f"Progress: {scraped_count} pages scraped, {len(urls_to_visit)} in queue")
        
        logger.info(f"Crawling finished. Total pages scraped: {scraped_count}")
        return scraped_count


def main():
    """Main function to run the crawler."""
    crawler = ManimDocsCrawler(
        start_url="https://docs.manim.community/en/stable/",
        output_dir="documents",
        delay=0.5,  # Be respectful with requests
        max_pages=None  # Set to a number to limit for testing
    )
    
    try:
        pages_scraped = crawler.crawl()
        print(f"\n✅ Successfully scraped {pages_scraped} pages!")
        print(f"📁 Content saved in '{crawler.output_dir}' directory")
        print("🚀 Ready to run VectorDB to create embeddings!")
        
    except KeyboardInterrupt:
        logger.info("Crawling interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during crawling: {e}")


if __name__ == "__main__":
    main() 