"""
Selenium automation script for downloading audio files from asmr.one.
Targets works with $age:general$ filter, downloads only "SEなし" (no SE) folders.
"""

import os
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BASE_URL = "https://asmr.one/works?keyword=%24age%3Ageneral%24"
WAIT_TIMEOUT = 15
PAGE_LOAD_WAIT = 3
DOWNLOAD_POPUP_WAIT = 10
DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
PROCESSED_FILE = "processed_ids.txt"


def js_click(driver, element):
    """Click element via JS dispatchEvent to bypass backdrop overlays."""
    driver.execute_script(
        "var evt = new MouseEvent('click', {bubbles: true, cancelable: true});"
        "arguments[0].dispatchEvent(evt);",
        element,
    )


# ---------------------------------------------------------------------------
# Processed IDs persistence
# ---------------------------------------------------------------------------

def load_processed_ids():
    """Load already-processed RJ IDs (numeric part only) from file."""
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_processed_id(rj_id: str):
    """Append a processed RJ ID (e.g. 'RJ01528180' -> '01528180') to file."""
    numeric = rj_id.replace("RJ", "")
    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(numeric + "\n")


def is_processed(rj_id: str, processed: set) -> bool:
    return rj_id.replace("RJ", "") in processed


# ---------------------------------------------------------------------------
# Browser setup
# ---------------------------------------------------------------------------

def create_driver():
    """Create Chrome driver with auto-download prefs."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": DOWNLOAD_DIR,
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    options.add_experimental_option("prefs", prefs)
    service = Service(executable_path="chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)
    driver.maximize_window()
    return driver


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------

def wait_for_works_grid(driver):
    """Wait until work cards are visible on the search page."""
    WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[id^='RJ']"))
    )
    time.sleep(1)  # let lazy-loaded cards render


def get_work_ids_on_page(driver) -> list[str]:
    """Return list of RJ IDs visible on the current page."""
    cards = driver.find_elements(By.CSS_SELECTOR, "div[id^='RJ']")
    return [c.get_attribute("id") for c in cards if c.get_attribute("id")]


def open_work_in_new_tab(driver, work_id: str):
    """Open a work detail page in a new tab and switch to it."""
    link = WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, f"a[href='/work/{work_id}']")
        )
    )
    # Ctrl+Click to open in new tab
    ActionChains(driver).key_down(Keys.CONTROL).click(link).key_up(Keys.CONTROL).perform()
    time.sleep(2)
    # Switch to the new tab (last handle)
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(PAGE_LOAD_WAIT)


def close_current_tab_and_switch_back(driver):
    """Close current tab and switch back to the first (main) tab."""
    if len(driver.window_handles) > 1:
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
    time.sleep(1)


# ---------------------------------------------------------------------------
# Download flow helpers
# ---------------------------------------------------------------------------

def click_download_button(driver):
    """Click the green Download button on the work detail page."""
    btn = WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "button.bg-green")
        )
    )
    js_click(driver, btn)
    log.info("Clicked page Download button")
    time.sleep(DOWNLOAD_POPUP_WAIT)  # wait for popup + tree to load


def uncheck_root_folder(driver):
    """Uncheck the root RJ folder checkbox to deselect everything.
    
    Targets the div.q-checkbox element (NOT the SVG inside it).
    """
    tree = WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, ".q-tree.q-tree--dense")
        )
    )
    # The first parent node is the root RJ folder
    root_node = tree.find_element(
        By.CSS_SELECTOR, ".q-tree__node--parent"
    )
    checkbox = root_node.find_element(
        By.CSS_SELECTOR, "div.q-checkbox"
    )
    js_click(driver, checkbox)
    log.info("Unchecked root folder")
    time.sleep(1)


def find_se_nashi_folder(driver):
    """Find the tree node whose folder name contains 'SEなし'.
    
    Returns the .q-tree__node--parent element, or None if not found.
    """
    tree = driver.find_element(By.CSS_SELECTOR, ".q-tree.q-tree--dense")
    parent_nodes = tree.find_elements(
        By.CSS_SELECTOR, ".q-tree__node--parent"
    )
    for node in parent_nodes:
        try:
            name_el = node.find_element(By.CSS_SELECTOR, ".col.ellipsis")
            name_text = name_el.text
            if "SEなし" in name_text:
                log.info(f"Found SEなし folder: {name_text}")
                return node
        except (NoSuchElementException, StaleElementReferenceException):
            continue
    return None


def check_folder_checkbox(driver, node):
    """Check the checkbox of a specific tree node.
    
    Targets div.q-checkbox (NOT the SVG).
    """
    checkbox = node.find_element(By.CSS_SELECTOR, "div.q-checkbox")
    js_click(driver, checkbox)
    log.info("Checked SEなし folder checkbox")
    time.sleep(1)


def click_popup_download_button(driver):
    """Click the Download button inside the popup dialog.
    
    Strategy: After clicking the page Download button, a popup appears with
    a q-dialog overlay. The popup also has a bg-green Download button.
    We find all visible bg-green buttons and click the LAST one (the popup one).
    We use js_click to bypass the q-dialog__backdrop.
    """
    time.sleep(2)  # ensure popup is fully rendered
    
    # Find all green buttons on the page
    green_buttons = driver.find_elements(By.CSS_SELECTOR, "button.bg-green")
    log.info(f"Found {len(green_buttons)} green button(s) total")
    
    if len(green_buttons) >= 2:
        # The popup button is the second/last one
        popup_btn = green_buttons[-1]
        log.info(f"Clicking popup download button (last of {len(green_buttons)})")
        js_click(driver, popup_btn)
    elif len(green_buttons) == 1:
        # Only one green button - try finding inside q-dialog specifically
        try:
            dialog_btn = driver.find_element(
                By.CSS_SELECTOR,
                "div.q-dialog button.bg-green"
            )
            log.info("Found download button inside q-dialog")
            js_click(driver, dialog_btn)
        except NoSuchElementException:
            # Fallback: click the only green button
            log.warning("Only 1 green button, no q-dialog button found. Clicking it anyway.")
            js_click(driver, green_buttons[0])
    else:
        log.error("No green buttons found at all!")
        return
    
    log.info("Popup download button clicked - download should start")
    time.sleep(3)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def go_to_next_page(driver) -> bool:
    """Click the 'next page' button. Returns False if on last page."""
    try:
        next_btn = driver.find_element(
            By.CSS_SELECTOR,
            "li.ant-pagination-next:not(.ant-pagination-disabled) a"
        )
        js_click(driver, next_btn)
        time.sleep(PAGE_LOAD_WAIT)
        return True
    except NoSuchElementException:
        log.info("No more pages (next button disabled or not found)")
        return False


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_work(driver, work_id: str) -> bool:
    """Process a single work: open tab, select SEなし, download, close tab.
    
    Returns True if download was initiated, False otherwise.
    """
    log.info(f"Processing {work_id}...")
    
    try:
        open_work_in_new_tab(driver, work_id)
        
        # Click the green Download button on the detail page
        click_download_button(driver)
        
        # Find SEなし folder in the popup tree
        se_nashi = find_se_nashi_folder(driver)
        if se_nashi is None:
            log.warning(f"{work_id}: No SEなし folder found, skipping")
            close_current_tab_and_switch_back(driver)
            return False
        
        # Uncheck root to deselect all, then check only SEなし
        uncheck_root_folder(driver)
        check_folder_checkbox(driver, se_nashi)
        
        # Click the download button in the popup
        click_popup_download_button(driver)
        
        log.info(f"{work_id}: Download initiated for SEなし folder")
        
        # Wait a bit for download to start before closing tab
        time.sleep(5)
        close_current_tab_and_switch_back(driver)
        return True
        
    except Exception as e:
        log.error(f"Error processing {work_id}: {e}")
        # Make sure we get back to the main tab
        try:
            close_current_tab_and_switch_back(driver)
        except Exception:
            pass
        return False


def main():
    processed = load_processed_ids()
    log.info(f"Loaded {len(processed)} already-processed IDs")
    
    driver = create_driver()
    
    try:
        driver.get(BASE_URL)
        time.sleep(PAGE_LOAD_WAIT)
        
        page_num = 1
        while True:
            log.info(f"=== Page {page_num} ===")
            wait_for_works_grid(driver)
            work_ids = get_work_ids_on_page(driver)
            log.info(f"Found {len(work_ids)} works on page {page_num}")
            
            for wid in work_ids:
                if is_processed(wid, processed):
                    log.info(f"{wid}: already processed, skipping")
                    continue
                
                success = process_work(driver, wid)
                if success:
                    save_processed_id(wid)
                    processed.add(wid.replace("RJ", ""))
                
                time.sleep(2)  # small delay between works
            
            # Go to next page
            if not go_to_next_page(driver):
                break
            page_num += 1
            wait_for_works_grid(driver)
    
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Fatal error: {e}")
    finally:
        log.info("Done. Closing browser.")
        driver.quit()


if __name__ == "__main__":
    main()
