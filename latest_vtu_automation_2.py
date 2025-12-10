#!/usr/bin/env python3
"""
Optimized VTU Results Automation Script
Ultra-fast synchronous implementation with local model captcha solving
Direct form URL access version
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError
# Import the new local CAPTCHA model service
from captcha_model_recognizer import recognize_captcha_with_model  # Use the new local model service

class VTUAutomation:
    def __init__(self, headless=True):
        self.results_dir = Path('results')
        self.captcha_dir = Path('captchas')
        self.pdf_dir = Path('pdf_results')
        self.results_dir.mkdir(exist_ok=True)
        self.captcha_dir.mkdir(exist_ok=True)
        self.pdf_dir.mkdir(exist_ok=True)
        self.current_batch_name = None
        
        print("✅ Local CAPTCHA model solver initialized")
        
        # Configuration (optimized for speed + reliability)
        self.max_captcha_retries = 5
        self.page_timeout = 15000  # Reduced from 30s → faster failover
        self.request_delay = 0.2  # Reduced from 2.2s → optimized for speed
        self.headless = headless   # Set to False only for debugging
        
    def _sanitize_filename_component(self, text):
        """Sanitize a string to be safely used as part of a filename."""
        if not text:
            return ""
        import re
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_.-")
        return sanitized or "batch"

    def process_single_usn(self, usn, form_url):
        """Process a single USN and return result"""
        result = {'success': False, 'files': [], 'error': None, 'usn': usn}
        
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=self.headless)
                result = self._process_with_browser(browser, usn, form_url)
                browser.close()
        except Exception as e:
            result['error'] = f"Browser error: {str(e)}"
            
        return result

    def _process_with_browser(self, browser, usn, form_url):
        """Main browser automation logic — SPEED OPTIMIZED"""
        result = {'success': False, 'files': [], 'error': None, 'usn': usn}
        page = None
        
        try:
            page = browser.new_page()
            page.set_default_timeout(self.page_timeout)
            
            print(f"🌐 Opening VTU results form for {usn}...")
            # Directly navigate to the form URL provided by user
            page.goto(form_url, wait_until="networkidle")
            
            # Verify we're on the correct page by checking for form elements
            try:
                page.wait_for_selector("input[name='lns']", timeout=10000)
                page.wait_for_selector("input[name='captchacode']", timeout=10000)
                print(f"✅ Navigated to form page: {form_url}")
            except TimeoutError:
                result['error'] = f"Form elements not found on page: {form_url}"
                return result
            
            # STEP 1: Fill USN and solve CAPTCHA in loop — MAX SPEED
            for attempt in range(1, self.max_captcha_retries + 1):
                print(f"🔍 CAPTCHA attempt {attempt}/{self.max_captcha_retries} for {usn}")
                
                # Clear and refill USN — no delays
                usn_input = page.locator("input[name='lns']")
                usn_input.fill("")  # Clear
                usn_input.fill(usn)
                
                # Screenshot CAPTCHA — use direct locator
                captcha_path = self.captcha_dir / f"captcha_{usn}_{attempt}.png"
                try:
                    captcha_img = page.locator("img[src*='captcha']").first
                    if not captcha_img.is_visible():
                        captcha_img = page.locator("img[alt='CAPTCHA']").first
                    if not captcha_img.is_visible():
                        captcha_img = page.locator("img").nth(2)
                    captcha_img.screenshot(path=str(captcha_path))
                except Exception as e:
                    print(f"⚠️ Could not capture CAPTCHA: {e}")
                    if attempt < self.max_captcha_retries:
                        continue
                    result['error'] = "Failed to capture CAPTCHA image"
                    return result
                
                # Solve CAPTCHA with local model service
                captcha_text = recognize_captcha_with_model(str(captcha_path))
                
                if not captcha_text or len(captcha_text) != 6:
                    print(f"⚠️ Invalid CAPTCHA length: '{captcha_text}' ({len(captcha_text) if captcha_text else 0})")
                    if attempt < self.max_captcha_retries:
                        continue
                    result['error'] = f"CAPTCHA solver returned invalid text: {captcha_text}"
                    return result
                
                print(f"🤖 Entering CAPTCHA: {captcha_text}")
                
                # Fill CAPTCHA and submit — NO DELAYS
                captcha_input = page.locator("input[name='captchacode']")
                captcha_input.fill(captcha_text)
                
                # Submit and WAIT FOR RESULT or ERROR — NO time.sleep()
                submit_button = page.get_by_role("button", name="ಸಲ್ಲಿಸಿ / SUBMIT")
                submit_button.click()
                
                # === CRITICAL SPEED OPTIMIZATION: WAIT FOR OUTCOME ===
                # Wait max 5s for either success OR error message
                try:
                    # Success: Result table visible
                    page.wait_for_selector("text=University Seat Number :", timeout=5000)
                    print("✅ SUCCESS! Results loaded.")
                    return self._save_results(page, usn)
                except TimeoutError:
                    pass
                
                try:
                    # Invalid USN
                    if page.locator("text=Invalid USN").is_visible(timeout=1000):
                        print("❌ Invalid USN or result not published")
                        result['error'] = "Invalid USN or result not published"
                        return result
                except TimeoutError:
                    pass
                
                try:
                    # CAPTCHA incorrect → form reappears
                    if page.locator("input[name='lns']").is_visible(timeout=1000) and page.locator("input[name='captchacode']").is_visible(timeout=1000):
                        print("❌ CAPTCHA incorrect — retrying...")
                        time.sleep(0.5)  # Tiny pause before next loop
                        continue
                except TimeoutError:
                    pass
                
                # Any other issue? Reload form page
                if "chrome-error" in page.url:
                    print("❌ Chrome error — reloading form...")
                    page.goto(form_url)
                    time.sleep(0.5)
                    continue
                
                # Unknown state — retry immediately
                print("❓ Unknown state — retrying...")
                time.sleep(0.5)

            # All retries failed
            result['error'] = f"Failed to solve CAPTCHA after {self.max_captcha_retries} attempts"
            return result
            
        except Exception as e:
            result['error'] = f"Processing error: {str(e)}"
            return result
        finally:
            if page is not None:
                page.close()

    def _save_results(self, page, usn):
        """Save results as PDF only — FAST"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_batch = self._sanitize_filename_component(self.current_batch_name) if self.current_batch_name else None
        batch_dir = self.pdf_dir / (safe_batch if safe_batch else "batch")
        batch_dir.mkdir(parents=True, exist_ok=True)
        pdf_filename = f"{usn}_{timestamp}.pdf"
        pdf_path = batch_dir / pdf_filename
        
        # Save PDF — only if page is stable
        try:
            page.pdf(path=str(pdf_path), format="A4", margin={"top": "10px", "bottom": "10px"})
            print(f"✅ Results saved: {pdf_path.name}")
        except Exception as e:
            print(f"⚠️ PDF generation failed: {e}")
            return {
                'success': False,
                'files': [],
                'error': f"PDF generation failed: {e}",
                'usn': usn
            }
        
        return {
            'success': True,
            'files': [
                {'filename': pdf_path.name, 'type': 'PDF', 'path': str(pdf_path)}
            ],
            'error': None,
            'usn': usn
        }

    def process_batch_with_updates(self, usns, form_url, batch_name="Batch"):
        """Process multiple USNs in sequence — WITH SPEED BOOST"""
        # Track current batch name for file naming
        self.current_batch_name = batch_name
        print(f"🚀 Starting batch: {batch_name} ({len(usns)} USNs)")
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
            print(f"\n--- Processing {i}/{len(usns)}: {usn} ---")
            
            result = self.process_single_usn(usn, form_url)
            batch_result['results'].append(result)
            
            if result['success']:
                batch_result['successful'] += 1
                print(f"✅ {usn}: Success")
            else:
                batch_result['failed'] += 1
                print(f"❌ {usn}: {result['error']}")
            
            # Delay between requests — minimal but safe
            if i < len(usns):
                print(f"⏳ Waiting {self.request_delay:.1f} seconds...")
                time.sleep(self.request_delay)
        
        batch_result['end_time'] = datetime.now().isoformat()
        
        # Save batch summary
        summary_path = self.results_dir / f"summary_{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_result, f, indent=2)
        
        success_rate = (batch_result['successful'] / batch_result['total'] * 100) if batch_result['total'] > 0 else 0
        print(f"\n🎉 Batch completed!")
        print(f"✅ Successful: {batch_result['successful']}")
        print(f"❌ Failed: {batch_result['failed']}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        print(f"📁 Summary: {summary_path}")
        
        return batch_result


def main():
    """Test function — set headless=True for speed"""
    automation = VTUAutomation(headless=True)  # 🔥 HEADLESS MODE FOR MAX SPEED
    
    # Test with real known-good USN (replace with valid one)
    test_usn = "1SP23AD027"
    form_url = "https://results.vtu.ac.in/JJEcbcs25/index.php"
    print(f"🧪 Testing with USN: {test_usn}")
    print(f"🔗 Using form URL: {form_url}")
    
    result = automation.process_single_usn(test_usn, form_url)
    
    if result['success']:
        print("🎉 Test successful!")
        for file in result['files']:
            print(f"📁 Generated: {file['filename']}")
    else:
        print(f"❌ Test failed: {result['error']}")