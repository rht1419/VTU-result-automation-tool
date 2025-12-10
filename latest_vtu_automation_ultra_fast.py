#!/usr/bin/env python3
"""
Ultra-Fast VTU Results Automation Script
Maximum speed optimization with aggressive performance tuning
Direct form URL access version
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError
from captcha_model_recognizer import recognize_captcha_with_model

class VTUAutomationUltraFast:
    def __init__(self, headless=False, enable_pdf=False):
        self.results_dir = Path('results')
        self.captcha_dir = Path('captchas')
        self.pdf_dir = Path('pdf_results')
        self.results_dir.mkdir(exist_ok=True)
        self.captcha_dir.mkdir(exist_ok=True)
        self.pdf_dir.mkdir(exist_ok=True)
        self.current_batch_name = None
        
        # Initialize local captcha solver
        try:
            print("✅ Local CAPTCHA model solver initialized")
        except Exception as e:
            print(f"❌ Failed to initialize CAPTCHA solver: {e}")
            raise Exception("CAPTCHA solver is required for automation")
        
        # ULTRA-FAST Configuration (maximum speed optimization)
        self.max_captcha_retries = 5
        self.page_timeout = 8000       # Reduced from 15s → 8s (faster failover)
        self.request_delay = 0.1       # Ultra-minimal delay (from 0.2s)
        self.element_timeout = 4000    # Faster element detection
        self.result_timeout = 2500     # Faster result detection
        self.headless = headless
        self.enable_pdf = enable_pdf   # PDF generation optional for speed

    def _get_browser_args(self):
        """Ultra-fast browser arguments for maximum performance"""
        return [
            '--disable-blink-features=AutomationControlled',
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--disable-extensions',
            '--disable-plugins',
            '--disable-images',  # Skip images for faster loading
            '--disable-background-networking',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-features=TranslateUI',
            '--disable-ipc-flooding-protection',
            '--disable-client-side-phishing-detection',
            '--disable-component-update',
            '--disable-default-apps',
            '--disable-domain-reliability',
            '--disable-features=AudioServiceOutOfProcess',
            '--disable-hang-monitor',
            '--disable-popup-blocking',
            '--disable-prompt-on-repost',
            '--disable-sync',
            '--disable-web-security',
            '--memory-pressure-off',
            '--max_old_space_size=4096',
            '--aggressive-cache-discard',
            '--window-size=1920,1080',
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
        ]

    def process_single_usn(self, usn, form_url):
        """Process a single USN with maximum speed optimization"""
        result = {'success': False, 'files': [], 'error': None, 'usn': usn}
        
        # Create a completely new Playwright instance for each USN
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=self.headless,
                    args=self._get_browser_args()
                )
                
                context = browser.new_context()
                result = self._process_with_browser_ultra_fast(context, usn, form_url)
                browser.close()
        except Exception as e:
            result['error'] = f"Browser error: {str(e)}"
            
        return result

    def _process_with_browser_ultra_fast(self, context, usn, form_url):
        """Ultra-fast browser automation with aggressive optimization"""
        result = {'success': False, 'files': [], 'error': None, 'usn': usn}
        page = None
        
        try:
            page = context.new_page()
            page.set_default_timeout(self.page_timeout)
            
            print(f"⚡ Ultra-fast processing for {usn}...")
            
            # STEP 1: Navigate directly to form URL with aggressive timeout
            page.goto(form_url, wait_until="domcontentloaded")  # Faster than networkidle
            
            # Verify we're on the correct page by checking for form elements
            try:
                page.wait_for_selector("input[name='lns']", timeout=self.element_timeout)
                page.wait_for_selector("input[name='captchacode']", timeout=self.element_timeout)
                print(f"✅ Ultra-fast navigation completed: {form_url}")
            except TimeoutError:
                result['error'] = f"Form elements not found on page: {form_url}"
                return result
            
            # STEP 2: Ultra-fast USN and CAPTCHA processing
            for attempt in range(1, self.max_captcha_retries + 1):
                print(f"⚡ Ultra-fast captcha attempt {attempt}/{self.max_captcha_retries} for {usn}")
                
                # Instant USN entry (already optimized from memory)
                usn_input = page.locator("input[name='lns']")
                usn_input.fill("")
                usn_input.fill(usn)
                
                # Fast CAPTCHA capture
                captcha_path = self.captcha_dir / f"captcha_{usn}_{attempt}.png"
                try:
                    captcha_img = page.locator("img[src*='captcha']").first
                    if not captcha_img.is_visible(timeout=1000):
                        captcha_img = page.locator("img[alt='CAPTCHA']").first
                    if not captcha_img.is_visible(timeout=1000):
                        captcha_img = page.locator("img").nth(2)
                    captcha_img.screenshot(path=str(captcha_path))
                except Exception as e:
                    print(f"⚠️ CAPTCHA capture failed: {e}")
                    if attempt < self.max_captcha_retries:
                        continue
                    result['error'] = "Failed to capture CAPTCHA image"
                    return result
                
                # Fast local model processing
                captcha_text = recognize_captcha_with_model(str(captcha_path))
                if captcha_text:
                    captcha_text = re.sub(r'[^a-zA-Z0-9]', '', captcha_text)
                else:
                    captcha_text = ""
                
                if len(captcha_text) != 6:
                    print(f"⚠️ Invalid CAPTCHA: '{captcha_text}'")
                    if attempt < self.max_captcha_retries:
                        continue
                    result['error'] = f"CAPTCHA validation failed: {captcha_text}"
                    return result
                
                print(f"🤖 Ultra-fast CAPTCHA entry: {captcha_text}")
                
                # Instant form submission
                captcha_input = page.locator("input[name='captchacode']")
                captcha_input.fill(captcha_text)
                
                submit_button = page.get_by_role("button", name="ಸಲ್ಲಿಸಿ / SUBMIT")
                submit_button.click()
                
                # ULTRA-FAST RESULT DETECTION with reduced timeouts
                try:
                    # Success detection - fastest timeout
                    page.wait_for_selector("text=University Seat Number :", timeout=self.result_timeout)
                    print("✅ ULTRA-FAST SUCCESS! Results loaded.")
                    return self._save_results_ultra_fast(page, usn)
                except TimeoutError:
                    pass
                
                # Fast error detection
                try:
                    if page.locator("text=Invalid USN").is_visible(timeout=500):
                        print("❌ Invalid USN detected")
                        result['error'] = "Invalid USN or result not published"
                        return result
                except TimeoutError:
                    pass
                
                # Fast retry detection
                try:
                    if page.locator("input[name='lns']").is_visible(timeout=500) and page.locator("input[name='captchacode']").is_visible(timeout=500):
                        print("❌ CAPTCHA incorrect - ultra-fast retry...")
                        time.sleep(0.1)  # Minimal retry delay
                        continue
                except TimeoutError:
                    pass
                
                # Chrome error handling
                if "chrome-error" in page.url:
                    print("❌ Chrome error - reloading...")
                    page.goto(form_url, wait_until="domcontentloaded")
                    time.sleep(0.1)
                    continue
                
                # Unknown state - minimal delay
                print("❓ Unknown state - retrying...")
                time.sleep(0.1)

            # All retries failed
            result['error'] = f"Failed to solve CAPTCHA after {self.max_captcha_retries} attempts"
            return result
            
        except Exception as e:
            result['error'] = f"Processing error: {str(e)}"
            return result
        finally:
            # Don't close browser - keep session alive for speed
            if page is not None:
                page.close()

    def _save_results_ultra_fast(self, page, usn):
        """Ultra-fast result saving with optional PDF only"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files = []
        
        # Optional PDF generation for speed
        if self.enable_pdf:
            safe_batch = self._sanitize_filename_component(self.current_batch_name) if self.current_batch_name else None
            batch_dir = self.pdf_dir / (safe_batch if safe_batch else "batch")
            batch_dir.mkdir(parents=True, exist_ok=True)
            pdf_filename = f"{usn}_{timestamp}.pdf"
            pdf_path = batch_dir / pdf_filename
            try:
                page.pdf(path=str(pdf_path), format="A4", margin={"top": "10px", "bottom": "10px"})
                files.append({'filename': pdf_path.name, 'type': 'PDF', 'path': str(pdf_path)})
                print(f"✅ Ultra-fast save: {pdf_path.name}")
            except Exception as e:
                print(f"⚠️ PDF generation failed: {e}")
        else:
            print(f"⏭️ Skipping PDF generation (disabled)")
        
        return {
            'success': True,
            'files': files,
            'error': None,
            'usn': usn
        }

    def process_batch_with_updates(self, usns, form_url, batch_name="Ultra-Fast Batch"):
        """Ultra-fast batch processing with maximum optimization"""
        # Track current batch name for file naming
        self.current_batch_name = batch_name
        print(f"🚀 Starting ULTRA-FAST batch: {batch_name} ({len(usns)} USNs)")
        print(f"🔗 Using form URL: {form_url}")
        
        batch_result = {
            'batch_name': batch_name,
            'form_url': form_url,
            'total': len(usns),
            'successful': 0,
            'failed': 0,
            'results': [],
            'start_time': datetime.now().isoformat()
        }
        
        for i, usn in enumerate(usns, 1):
            print(f"\n⚡ Ultra-fast processing {i}/{len(usns)}: {usn}")
            
            result = self.process_single_usn(usn, form_url)
            batch_result['results'].append(result)
            
            if result['success']:
                batch_result['successful'] += 1
                print(f"✅ {usn}: Ultra-fast success")
            else:
                batch_result['failed'] += 1
                print(f"❌ {usn}: {result['error']}")
            
            # Ultra-minimal delay between requests
            if i < len(usns):
                print(f"⚡ Ultra-fast delay: {self.request_delay:.1f}s...")
                time.sleep(self.request_delay)
        
        batch_result['end_time'] = datetime.now().isoformat()
        
        # Save batch summary
        summary_path = self.results_dir / f"summary_ultrafast_{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_result, f, indent=2)
        
        success_rate = (batch_result['successful'] / batch_result['total'] * 100) if batch_result['total'] > 0 else 0
        print(f"\n🎉 ULTRA-FAST batch completed!")
        print(f"✅ Successful: {batch_result['successful']}")
        print(f"❌ Failed: {batch_result['failed']}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        print(f"📁 Summary: {summary_path}")
        
        return batch_result

    def __del__(self):
        """Cleanup method"""
        pass

    def _sanitize_filename_component(self, text):
        """Sanitize a string to be safely used as part of a filename."""
        if not text:
            return ""
        # Replace non-alphanumeric characters with underscores and trim
        import re
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_.-")
        # Fallback if becomes empty
        return sanitized or "batch"


def main():
    """Ultra-fast test function"""
    automation = VTUAutomationUltraFast(headless=True, enable_pdf=True)  # 🔥 MAXIMUM SPEED
    
    # Test with USN
    test_usn = "1SP23AD027"
    form_url = "https://results.vtu.ac.in/JJEcbcs25/index.php"
    print(f"⚡ Ultra-fast testing with USN: {test_usn}")
    print(f"🔗 Using form URL: {form_url}")
    
    result = automation.process_single_usn(test_usn, form_url)
    
    if result['success']:
        print("🎉 Ultra-fast test successful!")
        for file in result['files']:
            print(f"📁 Generated: {file['filename']}")
    else:
        print(f"❌ Test failed: {result['error']}")