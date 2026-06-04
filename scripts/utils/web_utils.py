from typing import Union, List, Optional, Dict, Set
from numpy.typing import ArrayLike, NDArray
from typing import Sequence, Tuple, Union
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    StaleElementReferenceException,
    WebDriverException,
    JavascriptException
)
import os
import io
import time
from typing import Optional, Tuple
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
from .logger_utils import PhishLLMLogger
import torch.nn as nn
from functools import partial
import logging
from logging.handlers import RotatingFileHandler
from tldextract import tldextract

'''webdriver utils'''
def _enable_python_logging(log_path: str = "selenium-debug.log") -> None:
    # Root logger (console + rotating file)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s:%(name)s:%(message)s"
        )
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
    root.addHandler(fh)
    logging.getLogger("selenium").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)

def boot_driver(
    python_log_file: Optional[str] = "selenium-debug.log",
) -> WebDriver:
    if python_log_file:
        _enable_python_logging(python_log_file)
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")  # set resolution
    options.add_argument("--no-sandbox")  # (Linux) avoids sandbox issues
    options.add_argument("--disable-dev-shm-usage")  # Fixes shared memory errors
    options.add_argument("--disable-gpu")  # (Windows) GPU acceleration off in headless
    options.add_argument("--no-proxy-server")
    service = Service(
        ChromeDriverManager().install(),
    )
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def restart_driver(driver: WebDriver) -> WebDriver:
    driver.quit()
    time.sleep(2)
    return boot_driver()

def is_valid_domain(domain: Union[str, None]) -> bool:
    # Regular expression to check if the string is a valid domain without spaces
    if domain is None:
        return False
    pattern = re.compile(
        r'^(?!-)'  # Cannot start with a hyphen
        r'(?!.*--)'  # Cannot have two consecutive hyphens
        r'(?!.*\.\.)'  # Cannot have two consecutive periods
        r'(?!.*\s)'  # Cannot contain any spaces
        r'[a-zA-Z0-9-]{1,63}'  # Valid characters are alphanumeric and hyphen
        r'(?:\.[a-zA-Z]{2,})+$'  # Ends with a valid top-level domain
    )
    it_is_a_domain = bool(pattern.fullmatch(domain))
    return it_is_a_domain


# -- Robust domain extraction from free-form answers --
def normalize_domain(text: str) -> Optional[str]:
    """
    Extract and normalize a domain from model output.
    Accepts bare domains possibly wrapped with punctuation or code fences.
    Returns eTLD+1 style if valid, else None.
    """
    if not text:
        return None

    # Common cleanup: strip code fences/quotes and trailing punctuation
    s = text.strip().strip("`'\" \t\r\n;,:.()[]{}")
    s = s.replace("http://", "").replace("https://", "").replace("www.", "")
    s = s.split()[0]  # take the first token if multiple words

    # Prefer explicit domain-like substrings anywhere in the string
    candidates = re.findall(r'\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b', text)
    if s not in candidates:
        candidates = [s] + candidates

    for cand in candidates:
        cand = cand.strip().lower().strip("`'\" \t\r\n;,:.()[]{}")
        # Validate via tldextract + your is_valid_domain helper
        try:
            ext = tldextract.extract(cand)
            dom = '.'.join(p for p in (ext.domain, ext.suffix) if p)
        except Exception:
            continue
        if dom and is_valid_domain(dom):
            return dom
    return None

def url2logo(
    driver: WebDriver,
    url: str,
    logo_extractor: nn.Module
) -> Optional[Image.Image]:

    reference_logo = None
    try:
        driver.get(url)  # Visit the webpage
        time.sleep(2)
        screenshot_path = "tmp.png"
        driver.get_screenshot_as_file(screenshot_path)
        logo_boxes = logo_extractor(screenshot_path)
        if len(logo_boxes):
            logo_coord = logo_boxes[0]
            screenshot_img = Image.open(screenshot_path).convert("RGB")
            reference_logo = screenshot_img.crop((int(logo_coord[0]), int(logo_coord[1]),
                                                  int(logo_coord[2]), int(logo_coord[3])))
        os.remove(screenshot_path)
    except WebDriverException as e:
        print(f"Error accessing the webpage: {e}")
    except Exception as e:
        print(f"Failed to take screenshot: {e}")
    finally:
        driver = restart_driver(driver)
    return reference_logo


def query2url(
    query: str,
    SEARCH_ENGINE_API: str,
    SEARCH_ENGINE_ID: str,
    num: int = 10,
    proxies: Optional[Dict] = None
) -> List[str]:
    '''
        Google Search
    '''
    if len(query) == 0:
        return []

    num = int(num)
    URL = f"https://www.googleapis.com/customsearch/v1?key={SEARCH_ENGINE_API}&cx={SEARCH_ENGINE_ID}&q={query}&num={num}&filter=1"
    while True:
        try:
            data = requests.get(URL, proxies=proxies).json()
            break
        except requests.exceptions.SSLError as e:
            print(e)
            time.sleep(1)

    if data.get('error', {}).get('code') == 429:
        raise RuntimeError("Google search exceeds quota limit")

    search_items = data.get("items")
    if search_items is None:
        return []

    returned_urls = [item.get("link") for item in search_items]

    return returned_urls



def query2image(
    query: str,
    SEARCH_ENGINE_API: str,
    SEARCH_ENGINE_ID: str,
    num: int = 10,
    proxies: Optional[Dict] = None
) -> List[str]:
    '''
        Google Image Search
    '''
    if len(query) == 0:
        return []

    num = int(num)
    URL = f"https://www.googleapis.com/customsearch/v1?key={SEARCH_ENGINE_API}&cx={SEARCH_ENGINE_ID}&q={query}&searchType=image&num={num}&filter=1"
    while True:
        try:
            data = requests.get(URL, proxies=proxies).json()
            break
        except requests.exceptions.SSLError as e:
            print(e)
            time.sleep(1)

    if data.get('error', {}).get('code') == 429:
        raise RuntimeError("Google search exceeds quota limit")

    returned_urls = [
        item["image"]["thumbnailLink"]
        for item in data.get("items", [])
        if isinstance(item.get("image"), dict) and item["image"].get("thumbnailLink")
    ]

    return returned_urls


def download_image(
    url: str,
    proxies: Optional[Dict] = None
) -> Optional[Image.Image]:

    try:
        response = requests.get(url, proxies=proxies, timeout=5)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            return img
    except requests.exceptions.Timeout:
        print("Request timed out after", 5, "seconds.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading image: {e}")

    return None


def get_images(
    image_urls: List[str],
    proxies: Optional[Dict] = None
) -> List[Image.Image]:

    images = []
    if len(image_urls) > 0:
        with ThreadPoolExecutor(max_workers=len(image_urls)) as executor:
            futures = [executor.submit(download_image, url, proxies) for url in image_urls]
            for future in futures:
                img = future.result()
                if img:
                    images.append(img)

    return images


def is_alive_domain(
    domain: str,
    proxies: Optional[Dict] = None
) -> bool:
    try:
        response = requests.head('https://www.' + domain, timeout=10, proxies=proxies)  # Reduced timeout and used HEAD
        PhishLLMLogger.spit(f'Domain {domain}, status code {response.status_code}',
                            caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        if response.status_code < 400 or response.status_code in [405, 429] or response.status_code >= 500:
            PhishLLMLogger.spit(f'Domain {domain} is valid and alive', caller_prefix=PhishLLMLogger._caller_prefix,
                                debug=True)
            return True
        elif response.history and any([r.status_code < 400 for r in response.history]):
            PhishLLMLogger.spit(f'Domain {domain} is valid and alive', caller_prefix=PhishLLMLogger._caller_prefix,
                                debug=True)
            return True

    except Exception as err:
        PhishLLMLogger.spit(f'Error {err} when checking the aliveness of domain {domain}',
                            caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        return False

    PhishLLMLogger.spit(f'Domain {domain} is invalid or dead', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
    return False

def has_page_content_changed(
    curr_screenshot_elements: List[int],
    prev_screenshot_elements: List[int]
)-> bool:
    bincount_prev_elements = np.bincount(prev_screenshot_elements)
    bincount_curr_elements = np.bincount(curr_screenshot_elements)
    set_of_elements = min(len(bincount_prev_elements), len(bincount_curr_elements))
    screenshot_ele_change_ts = np.sum(
        bincount_prev_elements) // 2  # half the different UI elements distribution has changed

    if np.sum(np.abs(bincount_curr_elements[:set_of_elements] - bincount_prev_elements[
                                                                :set_of_elements])) > screenshot_ele_change_ts:
        PhishLLMLogger.spit(f"Webpage content has changed", caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        return True
    else:
        PhishLLMLogger.spit(f"Webpage content didn't change", caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        return False


def screenshot_element(
    elem: WebElement,
    dom: str,
    driver: WebDriver
) -> Tuple[Optional[str],
           Optional[Image.Image],
           Optional[str]]:
    """
    Returns:
        (candidate_ui, ele_screenshot_img, candidate_ui_text)
        - candidate_ui: the clickable_dom you passed in (or None on failure)
        - ele_screenshot_img: PIL.Image.Image of the element (or None on failure)
        - candidate_ui_text: element text/value (or None)
    """
    candidate_ui = None
    ele_screenshot_img = None
    candidate_ui_text = None

    try:
        # Scroll to top (plain Selenium)
        driver.execute_script("window.scrollTo(0, 0);")

        # Ensure the element is in view (center it to reduce cropping issues)
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center', inline:'center'});", elem)
        except Exception:
            pass

        # Basic visibility by rect
        rect = elem.rect  # {'x','y','width','height'} in CSS pixels
        w, h = rect.get("width", 0), rect.get("height", 0)
        if w <= 0 or h <= 0:
            return candidate_ui, ele_screenshot_img, candidate_ui_text

        # Preferred path: Selenium can screenshot elements directly
        try:
            png = elem.screenshot_as_png  # bytes
            ele_screenshot_img = Image.open(io.BytesIO(png))
            candidate_ui = dom
            etext = (elem.text or "")  # visible text
            if not etext:
                etext = elem.get_attribute("value") or ""
            candidate_ui_text = etext
            return candidate_ui, ele_screenshot_img, candidate_ui_text

        except (WebDriverException, StaleElementReferenceException):
            pass

        try:
            # Scroll offsets + device pixel ratio for accurate cropping
            sx, sy, dpr = driver.execute_script(
                "return [window.scrollX, window.scrollY, window.devicePixelRatio || 1];"
            )

            # Re-fetch rect in case it changed after scroll
            rect = elem.rect
            x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]

            # Convert page coords -> viewport coords, then scale by DPR
            left   = int((x - sx) * dpr)
            top    = int((y - sy) * dpr)
            right  = int((x - sx + w) * dpr)
            bottom = int((y - sy + h) * dpr)

            # Take a viewport screenshot and crop
            viewport_png = driver.get_screenshot_as_png()
            image = Image.open(io.BytesIO(viewport_png))

            # Clamp to image bounds
            left   = max(0, min(left,   image.width))
            top    = max(0, min(top,    image.height))
            right  = max(0, min(right,  image.width))
            bottom = max(0, min(bottom, image.height))

            if right > left and bottom > top:
                ele_screenshot_img = image.crop((left, top, right, bottom))
                candidate_ui = dom
                etext = (elem.text or "")
                if not etext:
                    etext = elem.get_attribute("value") or ""
                candidate_ui_text = etext

        except Exception as e2:
            print(f"Error processing element {dom} (crop fallback): {e2}")

    except Exception as e:
        print(f"Error accessing element {dom}: {e}")

    return candidate_ui, ele_screenshot_img, candidate_ui_text


def get_all_clickable_elements(
    driver: WebDriver
) -> Tuple[Tuple[List[WebElement], List[str]],
           Tuple[List[WebElement], List[str]],
           Tuple[List[WebElement], List[str]],
           Tuple[List[WebElement], List[str]]]:
    """
    Collect clickable elements using plain Selenium:
      - Buttons (<button>, input[type=button|submit|reset|image], role=button)
      - Links   (<a href>, role=link)
      - Images  (<img> that are inside links or have click handlers / button ancestors)
      - Leaf nodes (span/div/p/i without children) that look clickable
    Returns:
      (btns, btns_dom), (links, links_dom), (images, images_dom), (leaf_elements, leaf_elements_dom)
    """
    # Buckets
    btns, btns_dom = [], []
    links, links_dom = [], []
    images, images_dom = [], []
    leaf_elements, leaf_elements_dom = [], []

    # For deduping within each bucket
    seen_btn_xp = set()
    seen_link_xp = set()
    seen_img_xp = set()
    seen_leaf_xp = set()
    _REFS = {}

    # -------- helpers --------
    def register(ele: WebElement, dompath: str, bucket_list: List[WebElement], bucket_dom: List[str], seen_set: Set[str]):
        if not dompath or dompath in seen_set:
            return
        seen_set.add(dompath)
        bucket_list.append(ele)
        bucket_dom.append(dompath)
        try:
            # store callable with bound args
            _REFS[id(ele)] = (partial(driver.find_element, By.XPATH, dompath), (), {})
        except Exception:
            pass

    def safe_get_dompath(ele: WebElement) -> Optional[str]:
        js = r"""
            const el = arguments[0];
            if (!el) return null;
    
            // If it has an id, that's usually the most stable path.
            if (el.id) return `//*[@id="${el.id.replace(/"/g, '\\"')}"]`;
    
            function idx(node) {
              let i = 1, sib = node.previousElementSibling;
              while (sib) { if (sib.tagName === node.tagName) i++; sib = sib.previousElementSibling; }
              return i;
            }
    
            const parts = [];
            let node = el.nodeType === Node.ELEMENT_NODE ? el : el.parentElement;
            while (node && node.nodeType === Node.ELEMENT_NODE) {
              const tag = node.tagName.toLowerCase();
              // stop at html to keep the path reasonable
              if (tag === 'html') { parts.unshift('html'); break; }
              const position = idx(node);
              parts.unshift(`${tag}[${position}]`);
              node = node.parentElement;
            }
            return '//' + parts.join('/');
        """
        try:
            return driver.execute_script(js, ele)
        except (JavascriptException, StaleElementReferenceException, Exception):
            return None

    def is_clickable(ele: WebElement) -> bool:
        try:
            if ele is None:
                return False
            try:
                if not ele.is_enabled():
                    return False
            except WebDriverException:
                return False
            try:
                if not ele.is_displayed():
                    tabindex = (ele.get_attribute("tabindex") or "").strip()
                    if not (tabindex.lstrip("-").isdigit() and int(tabindex) >= 0):
                        return False
            except WebDriverException:
                return False

            tag = (ele.tag_name or "").lower()
            role = (ele.get_attribute("role") or "").lower()
            href = ele.get_attribute("href")
            onclick = ele.get_attribute("onclick")

            if tag == "a" and href:
                return True
            if tag == "button":
                return True
            if tag == "input":
                t = (ele.get_attribute("type") or "").lower()
                if t in {"button", "submit", "reset", "image"}:
                    return True

            if role in {"button", "link"}:
                return True

            # Cursor pointer heuristic (fixed to match the no-space version)
            style = (ele.get_attribute("style") or "").lower().replace(" ", "")
            if "cursor:pointer" in style:
                return True
            if onclick:
                return True

            # Ancestor heuristics (limit exceptions to Selenium ones)
            try:
                ele.find_element(By.XPATH, "ancestor::a[@href]")
                return True
            except (NoSuchElementException, StaleElementReferenceException):
                pass

            try:
                ele.find_element(By.XPATH, "ancestor::button[not(@disabled)] | ancestor::*[@role='button']")
                return True
            except (NoSuchElementException, StaleElementReferenceException):
                pass

            return False

        except (StaleElementReferenceException, WebDriverException):
            return False

    def safe_find_all(xpath: str) -> List[WebElement]:
        try:
            return driver.find_elements(By.XPATH, xpath) or []
        except Exception:
            return []

    # -------- buttons --------
    buttons_xpath = (
        "//button[not(@disabled)]"
        " | //input[(translate(@type,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')="
        "'button' or 'submit' or 'reset' or 'image') and not(@disabled)]"
        " | //*[@role='button' and not(@disabled)]"
    )
    for ele in safe_find_all(buttons_xpath):
        try:
            if not is_clickable(ele):
                continue
            xp = safe_get_dompath(ele)
            if xp:
                register(ele, xp, btns, btns_dom, seen_btn_xp)
        except (StaleElementReferenceException, WebDriverException):
            continue

    # -------- links --------
    links_xpath = (
        "//a[@href and not(@aria-disabled='true') and not(ancestor-or-self::*[@disabled])]"
    )
    for ele in safe_find_all(links_xpath):
        try:
            if not is_clickable(ele):
                continue
            xp = safe_get_dompath(ele)
            if xp:
                register(ele, xp, links, links_dom, seen_link_xp)
        except (StaleElementReferenceException, WebDriverException):
            continue

    # -------- clickable images --------
    images_xpath = (
        "//img[ancestor::a[@href] or @onclick or ancestor::*[@onclick] "
        " or ancestor::button[not(@disabled)] or ancestor::*[@role='button']]"
    )
    for ele in safe_find_all(images_xpath):
        try:
            if not is_clickable(ele):
                continue
            xp = safe_get_dompath(ele)
            if xp:
                register(ele, xp, images, images_dom, seen_img_xp)
        except (StaleElementReferenceException, WebDriverException):
            continue

    # -------- clickable leaf nodes --------
    # Limit to leafs; then filter to ones that "look clickable".
    leaf_xpath = (
        "//span[not(*)] | //div[not(*)] | //p[not(*)] | //i[not(*)]"
    )
    for ele in safe_find_all(leaf_xpath):
        try:
            if not is_clickable(ele):
                continue
            xp = safe_get_dompath(ele)
            if xp:
                register(ele, xp, leaf_elements, leaf_elements_dom, seen_leaf_xp)
        except (StaleElementReferenceException, WebDriverException):
            continue

    return (btns, btns_dom), (links, links_dom), (images, images_dom), (leaf_elements, leaf_elements_dom)


def page_transition(
    driver: WebDriver,
    dom: str,
    save_html_path: str,
    save_shot_path: str,
) -> Tuple[Optional[str],
           Optional[str],
           Optional[str],
           Optional[str]]:
    """
    Click an element (XPath = dom) and save updated screenshot + HTML.

    Returns:
        (etext, current_url, save_html_path, save_shot_path) or (None, None, None, None) on failure.
    """
    # --- Locate element ---
    try:
        elements = driver.find_elements(By.XPATH, dom)
        if not elements:
            print(f"No element found for XPath: {dom}")
            return None, None, None, None
        element = elements[0]
    except Exception as e:
        print(f"Exception {e} when locating element with XPath: {dom}")
        return None, None, None, None

    # --- Extract text/value and highlight target ---
    etext = None
    try:
        # best-effort text
        etext = (element.text or "") or (element.get_attribute("value") or "")
        if not etext:
            # fallbacks that sometimes carry labels
            etext = (element.get_attribute("aria-label") or "") or (element.get_attribute("placeholder") or "")
        # highlight
        try:
            driver.execute_script("arguments[0].style.outline='3px solid red';", element)
        except Exception:
            pass
    except StaleElementReferenceException:
        etext = None

    # --- Scroll into view & click ---
    try:
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center', inline:'center'});", element)
        except Exception:
            pass

        # Ensure it is clickable
        try:
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, dom)))
        except TimeoutException:
            # proceed anyway; sometimes custom overlays confuse EC
            pass

        old_url = driver.current_url
        old_source = driver.page_source

        # Move and click
        try:
            ActionChains(driver).move_to_element(element).pause(0.1).click(element).perform()
        except (StaleElementReferenceException, WebDriverException):
            # element may have gone stale; try to refetch once
            element = driver.find_elements(By.XPATH, dom)[0]
            ActionChains(driver).move_to_element(element).pause(0.1).click(element).perform()

        # Wait for navigation or DOM update (SPA-friendly)
        try:
            WebDriverWait(driver, 15).until(
                lambda d: d.current_url != old_url or d.page_source != old_source
            )
        except TimeoutException:
            # last resort: small sleep if the page is sluggish
            time.sleep(2)

        current_url = driver.current_url
    except Exception as e:
        print(f"Exception {e} when clicking the element")
        return None, None, None, None

    # --- Inject CSS to force black placeholders (optional) ---
    try:
        css_script = """
            var style = document.createElement('style');
            style.setAttribute('data-added-by', 'page_transition');
            style.appendChild(document.createTextNode(`
              ::-webkit-input-placeholder { color: black !important; }
              ::-moz-placeholder { opacity: 1; color: black !important; }
              :-ms-input-placeholder { color: black !important; }
              ::placeholder { color: black !important; }
            `));
            document.head && document.head.appendChild(style);
        """
        driver.execute_script(css_script)
    except Exception as e:
        print(f"CSS injection error: {e}")

    # --- Save screenshot + HTML ---
    try:
        # ensure parent dirs exist
        if save_shot_path:
            os.makedirs(os.path.dirname(save_shot_path) or ".", exist_ok=True)
        if save_html_path:
            os.makedirs(os.path.dirname(save_html_path) or ".", exist_ok=True)

        ok = driver.save_screenshot(save_shot_path)
        if not ok:
            print("Warning: driver.save_screenshot returned False")

        with open(save_html_path, "w", encoding="utf-8") as f:
            f.write(driver.page_source)

        print("Transition successful. New screenshot and HTML saved.")
        return etext, current_url, save_html_path, save_shot_path
    except Exception as e:
        print(f"Exception {e} when saving screenshot/HTML")
        return None, None, None, None