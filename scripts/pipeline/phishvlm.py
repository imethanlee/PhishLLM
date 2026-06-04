from openai import OpenAI
import torch
from torch import Tensor  # alias for torch.Tensor
from scripts.utils.utils import *
from scripts.utils.web_utils import *
from scripts.utils.draw_utils import draw_annotated_image_box
import os
from typing import List, Tuple, Optional, Union, Dict, Literal
import PIL
import json
from tldextract import tldextract
import urllib3
from urllib3.exceptions import MaxRetryError
import time, json, re, ast
from concurrent.futures import ThreadPoolExecutor, as_completed
urllib3.disable_warnings()
http = urllib3.PoolManager(maxsize=10)  # Increase the maxsize to a larger value, e.g., 10

_OPENAI_KEY_PATH = './datasets/openai_key.txt'
if not os.path.exists(_OPENAI_KEY_PATH):
    raise FileNotFoundError(
        f"OpenAI API key file not found at '{_OPENAI_KEY_PATH}'. "
        "Create it and paste your key inside (see the Setup section of README.md)."
    )
os.environ['OPENAI_API_KEY'] = open(_OPENAI_KEY_PATH).read().strip()
os.environ['CURL_CA_BUNDLE'] = ''


class PhishVLM():

    def __init__(
            self,
            logo_encoder: nn.Module,
            logo_extractor: nn.Module,
            layout_extractor: nn.Module,
            param_dict: Dict,
            proxies:Union[float, Dict] = None
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proxies = proxies
        self.logo_encoder   = logo_encoder
        self.logo_extractor = logo_extractor
        self.layout_extractor = layout_extractor

        ## LLM
        self.VLM_model = param_dict["VLM_model"]
        self.brand_prompt = param_dict['brand_recog']['prompt_path']
        self.crp_prompt = param_dict['crp_pred']['prompt_path']
        self.rank_prompt = param_dict['rank']['prompt_path']
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Load the Google API key and SEARCH_ENGINE_ID once during initialization
        self.API_KEY, self.SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]

        ## Load hyperparameters
        self.brand_recog_temperature, self.brand_recog_max_tokens = param_dict['brand_recog']['temperature'], param_dict['brand_recog']['max_tokens']
        self.brand_recog_sleep = param_dict['brand_recog']['sleep_time']
        self.do_brand_validation = param_dict['brand_valid']['activate']
        self.brand_valid_k, self.brand_valid_siamese_thre = param_dict['brand_valid']['k'], param_dict['brand_valid']['siamese_thre']

        self.crp_temperature, self.crp_max_tokens = param_dict['crp_pred']['temperature'], param_dict['crp_pred']['max_tokens']
        self.crp_sleep = param_dict['crp_pred']['sleep_time']

        self.rank_max_uis = param_dict['rank']['max_uis_process']
        self.rank_temperature, self.rank_max_tokens = param_dict['rank']['temperature'], param_dict['rank']['max_tokens']
        self.rank_driver_sleep = param_dict['rank']['driver_sleep_time']
        self.rank_driver_script_timeout = param_dict['rank']['script_timeout']
        self.rank_driver_page_load_timeout = param_dict['rank']['page_load_timeout']
        self.interaction_limit = param_dict['rank']['depth_limit']

        # webhosting domains as blacklist
        self.webhosting_domains = [x.strip() for x in open('./datasets/hosting_blacklists.txt').readlines()]


    def detect_logo(
        self,
        save_shot_path: str
    ) -> Tuple[Optional[List[float]], Optional[Image.Image]]:
        '''
            Logo detection
            Args: save_shot_path:
            Returns: (logo_box, reference_logo)
        '''
        reference_logo = None
        logo_box = None

        try:
            screenshot_img = Image.open(save_shot_path).convert("RGB")
            logo_boxes = self.logo_extractor(save_shot_path)
            if len(logo_boxes) > 0:
                logo_box = logo_boxes[0]  # get coordinate for logo
                reference_logo = screenshot_img.crop((int(logo_box[0]), int(logo_box[1]),
                                                      int(logo_box[2]), int(logo_box[3])))
        except PIL.UnidentifiedImageError:
            pass

        return logo_box, reference_logo


    def brand_recognition_llm(
        self,
        reference_logo: Optional[Image.Image]
    ) -> Tuple[Optional[str], Optional[Image.Image], float]:
        '''
            Brand Recognition Model
            Args:
             reference_logo:
             webpage_text:
             logo_caption:
             logo_ocr:
             announcer:
            Returns:
                (company_domain, company_logo, brand_llm_pred_time)

        '''
        company_domain: Optional[str] = None
        company_logo: Optional[Image.Image] = None
        brand_llm_pred_time: float = 0.0

        if not reference_logo:
            return company_domain, company_logo, brand_llm_pred_time

        # -- Load system prompt safely; provide a strict fallback that asks for a bare domain --
        try:
            with open(self.brand_prompt, 'r') as file:
                system_prompt = json.load(file)
        except Exception as e:
            PhishLLMLogger.spit(f"brand_prompt load failed: {e}", debug=True,
                                caller_prefix=PhishLLMLogger._caller_prefix)
            system_prompt = [{
                "role": "system",
                "content": (
                    "You are a vision-language assistant. Given a brand logo image, reply with only the brand’s "
                    "official primary website domain (e.g., 'microsoft.com'). Do not include any extra words, "
                    "protocols, slashes, or punctuation. If unsure, reply with just 'unknown'."
                )
            }]

        # Ensure RGB to avoid mode issues in downstream encoders
        try:
            logo_rgb = reference_logo.convert("RGB")
        except Exception:
            logo_rgb = reference_logo

        question = vlm_question_template_brand(logo_rgb)
        system_prompt.append(question)

        # -- Bounded retries with gentle backoff; keep original prompt-halving behavior on failure --
        max_retries = max(1, int(getattr(self, "brand_recog_max_retries", 3)))
        response = None

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.VLM_model,
                    messages=system_prompt,
                    temperature=getattr(self, "brand_recog_temperature", 0.0),
                    max_tokens=getattr(self, "brand_recog_max_tokens", 32),
                )
                brand_llm_pred_time = time.time() - start_time
                break
            except Exception as e:
                PhishLLMLogger.spit(f'LLM Exception {e}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                try:
                    last = system_prompt[-1]
                    if isinstance(last, dict) and isinstance(last.get('content'), str) and len(last['content']) > 0:
                        last['content'] = last['content'][:len(last['content']) // 2]
                except Exception:
                    pass

        answer = ''
        if response and getattr(response, "choices", None):
            answer = ''.join([getattr(choice.message, "content", "") for choice in response.choices]).strip()

        PhishLLMLogger.spit(f"Time taken for LLM brand prediction: {brand_llm_pred_time}\tDetected brand: {answer}",
                            debug=True,
                            caller_prefix=PhishLLMLogger._caller_prefix)

        parsed_domain = normalize_domain(answer)

        if parsed_domain:
            company_domain = parsed_domain
            company_logo = reference_logo  # preserve original behavior
            PhishLLMLogger.spit(
                f"Brand domain accepted: {company_domain}",
                debug=True,
                caller_prefix=PhishLLMLogger._caller_prefix
            )
        else:
            PhishLLMLogger.spit(
                "No valid domain found in LLM answer; leaving brand unknown.",
                debug=True,
                caller_prefix=PhishLLMLogger._caller_prefix
            )

        return company_domain, company_logo, brand_llm_pred_time

    def popularity_validation(
            self,
            company_domain: str
    ) -> Tuple[bool, float]:
        '''
            Brand recognition model : result validation
            Args:
             company_domain:
            Returns:
                (validation_success, searching_time)
        '''
        validation_success = False

        def _registrable(d: str) -> str:
            ext = tldextract.extract(d)
            dom = '.'.join(p for p in (ext.domain, ext.suffix) if p)
            return dom.lower()

        brand_reg = _registrable(company_domain)

        start_time = time.time()
        try:
            returned_urls = query2url(
                query=company_domain,
                SEARCH_ENGINE_ID=self.SEARCH_ENGINE_ID,
                SEARCH_ENGINE_API=self.API_KEY,
                num=getattr(self, "brand_valid_k", 10),
                proxies=getattr(self, "proxies", None),
            )
        except Exception as e:
            PhishLLMLogger.spit(f"query2url failed: {e}", debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
            returned_urls = []
        searching_time = time.time() - start_time

        # Extract registrable domains from results; dedupe and normalize away common prefixes like "www".
        returned_domains = set()
        for url in returned_urls:
            try:
                ext = tldextract.extract(url)
                dom = '.'.join(p for p in (ext.domain, ext.suffix) if p).lower()
                if dom:
                    returned_domains.add(dom)
            except Exception:
                continue

        # Success if the registrable brand domain appears among top results.
        validation_success = brand_reg in returned_domains

        return validation_success, searching_time

    def brand_validation(
            self,
            company_domain: str,
            reference_logo: Image.Image
    ) -> Tuple[bool, float, float]:
        '''
            Brand recognition model : result validation
            Args:
                company_domain
                reference_logo
            Returns:
                (validation_success, logo_searching_time, logo_matching_time)
        '''
        logo_searching_time, logo_matching_time = 0.0, 0.0
        validation_success = False

        if not reference_logo:
            return True, logo_searching_time, logo_matching_time

        # 1) Search candidate brand logos
        start_time = time.time()
        try:
            returned_urls = query2image(
                query=f'Brand: {company_domain} logo',
                SEARCH_ENGINE_ID=self.SEARCH_ENGINE_ID,
                SEARCH_ENGINE_API=self.API_KEY,
                num=getattr(self, "brand_valid_k", 10),
                proxies=getattr(self, "proxies", None),
            ) or []
        except Exception as e:
            PhishLLMLogger.spit(f"query2image failed: {e}", debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
            returned_urls = []
        logo_searching_time = time.time() - start_time

        try:
            logos = get_images(returned_urls, proxies=getattr(self, "proxies", None)) or []
        except Exception as e:
            PhishLLMLogger.spit(f"get_images failed: {e}", debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
            logos = []

        msg = f'Number of logos found on google images {len(logos)}'
        PhishLLMLogger.spit(msg, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

        if len(logos):
            try:
                reference_logo_feat = self.logo_encoder(reference_logo)
            except Exception as e:
                PhishLLMLogger.spit(f"logo_encoder(ref) failed: {e}", debug=True,
                                    caller_prefix=PhishLLMLogger._caller_prefix)
                return False, logo_searching_time, logo_matching_time

            start_time = time.time()
            sim_list: List[float] = []

            # Bound worker count to avoid resource spikes
            max_workers = max(1, min(8, len(logos)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.logo_encoder, logo): i for i, logo in enumerate(logos)}
                for fut in as_completed(futures):
                    try:
                        logo_feat = fut.result()
                        # Keep original similarity semantics (@). Cast to float if possible.
                        try:
                            matched_sim = float(reference_logo_feat @ logo_feat)  # type: ignore[operator]
                        except Exception:
                            # Fallback: try cosine-like similarity if @ fails
                            try:
                                num = (reference_logo_feat * logo_feat).sum()
                                den = (reference_logo_feat ** 2).sum() ** 0.5 * (logo_feat ** 2).sum() ** 0.5
                                matched_sim = float(num / max(den, 1e-8))
                            except Exception:
                                continue
                        sim_list.append(matched_sim)
                    except Exception as e:
                        PhishLLMLogger.spit(f"logo_encoder(candidate) failed: {e}", debug=True,
                                            caller_prefix=PhishLLMLogger._caller_prefix)
                        continue

            thr = getattr(self, "brand_valid_siamese_thre", 0.8)
            if any(s > thr for s in sim_list):
                validation_success = True

            logo_matching_time = time.time() - start_time

        return validation_success, logo_searching_time, logo_matching_time

    def crp_prediction_llm(
            self,
            webpage_screenshot: Image.Image
    ) -> Tuple[bool, float]:
        '''
            Use LLM to classify credential-requiring page v.s. non-credential-requiring page
            Args:
                webpage_screenshot:
            Returns:
                 (is_crp, inference_time_seconds)
        '''
        crp_llm_pred_time: float = 0.0

        # -- Load prompt safely --
        try:
            with open(self.crp_prompt, 'r') as file:
                system_prompt = json.load(file)
        except Exception as e:
            PhishLLMLogger.spit(f'Prompt load failed: {e}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
            system_prompt = [{
                "role": "system",
                "content": (
                    "You are a careful vision-language assistant. "
                    "Answer with a single letter: 'A' if the page requests credentials "
                    "(e.g., login/password/2FA), or 'B' if it does not."
                )
            }]

        # -- Be tolerant to image mode quirks --
        try:
            screenshot_rgb = webpage_screenshot.convert("RGB")
        except Exception:
            screenshot_rgb = webpage_screenshot  # fall back

        question = vlm_question_template_prediction(screenshot_rgb)
        system_prompt.append(question)

        # -- Bounded retry with gentle backoff; preserve original behavior of halving long prompts --
        max_retries = max(1, int(getattr(self, "crp_max_retries", 3)))
        response = None

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.VLM_model,
                    messages=system_prompt,
                    temperature=getattr(self, "crp_temperature", 0.0),
                    max_tokens=getattr(self, "crp_max_tokens", 32),
                )
                crp_llm_pred_time = time.time() - start_time
                break
            except Exception as e:
                PhishLLMLogger.spit(f'LLM Exception {e}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                # maybe the prompt is too long, cut by half (only if last message is plain text)
                try:
                    last = system_prompt[-1]
                    if isinstance(last, dict) and isinstance(last.get('content'), str) and len(last['content']) > 0:
                        last['content'] = last['content'][:len(last['content']) // 2]
                except Exception:
                    pass

        # -- Extract raw answer text robustly --
        answer = ''
        if response and getattr(response, "choices", None):
            answer = ''.join([getattr(choice.message, "content", "") for choice in response.choices]).strip()

        PhishLLMLogger.spit(f'Time taken for LLM CRP classification: {crp_llm_pred_time} \t CRP prediction: {answer}',
                            debug=True,
                            caller_prefix=PhishLLMLogger._caller_prefix)
        if 'A.' in answer:
            return True, crp_llm_pred_time  # CRP
        else:
            return False, crp_llm_pred_time

    def ranking_model(
            self,
            url: str,
            driver: WebDriver,
            ranking_model_refresh_page: bool,
    ) -> Tuple[Union[Sequence[str], str], Sequence[Tensor], WebDriver, float]:
        """
        Returns:
            candidate_uis_selected, candidate_imgs_selected, driver, transition_pred_time
        """
        transition_pred_time: float = 0.0

        # -- (Re)load page if needed
        if ranking_model_refresh_page:
            try:
                driver.get(url)
                time.sleep(getattr(self, "rank_driver_sleep", 0))
            except Exception as e:
                PhishLLMLogger.spit(e, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                driver = restart_driver(driver)
                try:
                    driver.get(url)
                    time.sleep(getattr(self, "rank_driver_sleep", 0))
                except Exception as e2:
                    PhishLLMLogger.spit(e2, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                    return [], [], driver, transition_pred_time

        # -- Collect clickables
        try:
            (btns, btns_dom), (links, links_dom), (images, images_dom), (others, others_dom) = get_all_clickable_elements(driver)
        except Exception as e:
            PhishLLMLogger.spit(e, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            return [], [], driver, transition_pred_time

        all_clickable     = btns + links + images + others
        all_clickable_dom = btns_dom + links_dom + images_dom + others_dom

        # -- Element screenshots
        candidate_uis:      List[Any] = []
        candidate_uis_imgs: List[Any] = []
        candidate_uis_text: List[str] = []

        max_uis = min(getattr(self, "rank_max_uis", 32), len(all_clickable))
        for it in range(max_uis):
            try:
                candidate_ui, candidate_ui_img, candidate_ui_text = screenshot_element(
                    all_clickable[it], all_clickable_dom[it], driver
                )
            except (MaxRetryError, WebDriverException, TimeoutException) as e:
                PhishLLMLogger.spit(e, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                driver = restart_driver(driver)
                continue
            except Exception as e:
                PhishLLMLogger.spit(e, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                continue

            if (candidate_ui is not None) and (candidate_ui_img is not None) and (candidate_ui_text is not None):
                candidate_uis.append(candidate_ui)
                candidate_uis_imgs.append(candidate_ui_img)
                candidate_uis_text.append(candidate_ui_text)

        # -- Rank them
        if len(candidate_uis_imgs):
            PhishLLMLogger.spit(f'Find {len(candidate_uis_imgs)} candidate UIs',
                                caller_prefix=PhishLLMLogger._caller_prefix, debug=True)

            # Heuristic: credential-taking keywords
            pattern = re.compile(Regexes.CREDENTIAL_TAKING_KEYWORDS, re.IGNORECASE | re.VERBOSE)
            indices = [i for i, text in enumerate(candidate_uis_text) if text and pattern.search(text)]

            if len(indices) > 0:
                candidate_uis_selected = [candidate_uis[ind] for ind in indices]
                candidate_imgs_selected = [candidate_uis_imgs[ind] for ind in indices]
                return candidate_uis_selected, candidate_imgs_selected, driver, transition_pred_time

            # VLM fallback
            try:
                with open(self.rank_prompt, 'r') as file:
                    system_prompt = json.load(file)
            except Exception as e:
                PhishLLMLogger.spit(f"Prompt load failed: {e}", caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                system_prompt = [{"role": "system", "content": "You are a careful vision-language assistant."}]

            question = vlm_question_template_transition(candidate_uis_imgs, candidate_uis_text)
            system_prompt.append(question)

            max_retries = max(1, int(getattr(self, "rank_max_retries", 3)))
            response = None
            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    response = self.client.chat.completions.create(
                        model=self.VLM_model,
                        messages=system_prompt,
                        temperature=getattr(self, "rank_temperature", 0.0),
                        max_tokens=getattr(self, "rank_max_tokens", 128),
                    )
                    transition_pred_time = time.time() - start_time
                    break
                except Exception as e:
                    PhishLLMLogger.spit(f'LLM Exception {e}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                    # shrink last message content if too long
                    try:
                        system_prompt[-1]['content'] = system_prompt[-1]['content'][:len(system_prompt[-1]['content']) // 2]
                    except Exception:
                        pass
                    time.sleep(getattr(self, "crp_sleep", 0))

            if not response or not getattr(response, "choices", None):
                return [], [], driver, transition_pred_time

            answer = ''.join([choice.message.content for choice in response.choices]) if response.choices else ""

            # -- Safe parse indices (replace eval)
            def _parse_indices(ans: str, n: int) -> List[int]:
                # try literal list first
                try:
                    parsed = ast.literal_eval(ans)
                    if isinstance(parsed, (list, tuple)):
                        ints = [int(x) for x in parsed if isinstance(x, (int, float, str)) and str(x).isdigit()]
                        return sorted({i for i in ints if 0 <= i < n})
                except Exception:
                    pass
                # fallback: extract all integers
                ints = [int(m.group()) for m in re.finditer(r"\d+", ans)]
                return sorted({i for i in ints if 0 <= i < n})

            indices = _parse_indices(answer, len(candidate_uis_imgs))

            if len(indices) > 0:
                candidate_uis_selected = [candidate_uis[ind] for ind in indices]
                candidate_imgs_selected = [candidate_uis_imgs[ind] for ind in indices]
                return candidate_uis_selected, candidate_imgs_selected, driver, transition_pred_time

            return [], [], driver, transition_pred_time

        return [], [], driver, transition_pred_time

    def test(
            self,
            url: str,
            reference_logo: Optional[Image.Image],
            logo_box: Optional[Sequence[float]],
            shot_path: str,
            html_path: str,
            driver: Optional[WebDriver] = None,
            limit: int = 0,
            brand_recog_time: float = 0.0,
            crp_prediction_time: float = 0.0,
            clip_prediction_time: float = 0.0,
            ranking_model_refresh_page: bool = True,
            skip_brand_recognition: bool = False,
            company_domain: Optional[str] = None,
            company_logo: Optional[Image.Image] = None,
    ) -> Tuple[Literal["phish", "benign"], str, float, float, float, Image.Image]:
        """
        PhishLLM
        Args:
            url: Current page URL.
            reference_logo: Detected/known logo image for the candidate brand on this page.
            logo_box: Bounding box of the detected logo.
            shot_path: Path to the current screenshot (PNG).
            html_path: Path to the current HTML snapshot.
            driver: Selenium WebDriver (may be reused across transitions).
            limit: Interaction depth so far (click depth).
            brand_recog_time: Accumulated time spent in brand recognition & validation.
            crp_prediction_time: Accumulated time spent in CRP prediction.
            clip_prediction_time: Accumulated time spent in ranking (CLIP) model.
            ranking_model_refresh_page: Whether last click changed the page content/URL.
            skip_brand_recognition: Skip brand recognition after the first hop.
            company_domain: Domain guessed/validated for the brand (if already known).
            company_logo: Canonical logo image for the brand (if already known).

        Returns:
            (label, target_brand, brand_time, crp_time, clip_time, annotated_image)
        """
        try:
            screenshot_img = Image.open(shot_path).convert("RGB")
        except Exception as e:
            PhishLLMLogger.spit(f"[!] Failed to open screenshot '{shot_path}': {e}", debug=True)
            # On I/O failure, err on the safe side and return benign with a blank image fallback
            screenshot_img = Image.new("RGB", (800, 600), "white")

        # -- Brand recognition (first page only unless forced) --
        if not skip_brand_recognition:
            company_domain, company_logo, br_time = self.brand_recognition_llm(reference_logo=reference_logo)
            brand_recog_time += br_time
            if getattr(self, "brand_recog_sleep", 0) > 0:
                time.sleep(self.brand_recog_sleep)

        # -- Hosting provider whitelist short-circuit --
        # If the predicted brand is itself a hosting/cloud provider, treat as benign.
        if company_domain and (company_domain in getattr(self, "webhosting_domains", set())):
            msg = '[\U00002705] Benign, since it is a brand providing cloud services'
            PhishLLMLogger.spit(msg)
            return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, screenshot_img

        # -- Domain-brand consistency check (only if we have a brand domain) --
        domain_brand_inconsistent = False
        if company_domain:
            # Extract domain parts once (avoid repeated parsing)
            pred_parts = tldextract.extract(company_domain)
            url_parts = tldextract.extract(url)
            domain4pred, suffix4pred = pred_parts.domain, pred_parts.suffix
            domain4url, suffix4url = url_parts.domain, url_parts.suffix
            # Inconsistency means suspicious (brand ≠ site)
            domain_brand_inconsistent = (domain4pred != domain4url) or (suffix4pred != suffix4url)

        phish_condition = domain_brand_inconsistent

        # Brand prediction results validation
        if phish_condition and (not skip_brand_recognition):
            if getattr(self, "do_brand_validation", False):
                # Validate by matching on-page logo to search results
                validation_success, logo_searching_time, logo_matching_time = self.brand_validation(
                    company_domain=company_domain,
                    reference_logo=reference_logo
                )
                brand_recog_time += (logo_searching_time + logo_matching_time)
                phish_condition = validation_success  # keep original behavior
                msg = (f"Time taken for brand validation (logo matching with Google Image search results): "
                       f"{logo_searching_time + logo_matching_time}<br>"
                       f"Domain {company_domain} is relevant and valid? {validation_success}")
                PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            else:
                # Simpler fallback: check if brand domain is alive
                try:
                    is_alive = is_alive_domain(f"{domain4pred}.{suffix4pred}", getattr(self, "proxies", None))
                except Exception as e:
                    is_alive = False
                    PhishLLMLogger.spit(f"[!] Brand validation (alive check) failed: {e}", debug=True)
                phish_condition = is_alive  # keep original behavior
                msg = f"Brand Validation: Domain {company_domain} is alive? {is_alive}"
                PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)

        if phish_condition:
            crp_cls, crp_time = self.crp_prediction_llm(webpage_screenshot=screenshot_img)
            crp_prediction_time += crp_time
            if getattr(self, "crp_sleep", 0) > 0:
                time.sleep(self.crp_sleep)

            if crp_cls:
                # CRP page detected -> Phish
                annotated = draw_annotated_image_box(screenshot_img, company_domain, logo_box)
                msg = f'[\u2757\uFE0F] Phishing discovered, phishing target is {company_domain}'
                PhishLLMLogger.spit(msg)
                return 'phish', company_domain, brand_recog_time, crp_prediction_time, clip_prediction_time, annotated

            # -- Not CRP: attempt a safe transition via ranking model (bounded by interaction limit) --
            if limit >= getattr(self, "interaction_limit", 0):
                msg = '[\U00002705] Benign, reached interaction limit ...'
                PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, screenshot_img

            # Ranking model
            candidate_elements, _, driver, clip_time = self.ranking_model(
                url=url,
                driver=driver,
                ranking_model_refresh_page=ranking_model_refresh_page
            )
            clip_prediction_time += clip_time

            if len(candidate_elements):
                # Prepare next snapshot paths
                save_html_path = re.sub(r"index[0-9]?\.html", f"index{limit}.html", html_path)
                save_shot_path = re.sub(r"shot[0-9]?\.png", f"shot{limit}.png", shot_path)

                # Choose which candidate to click
                if not ranking_model_refresh_page:
                    # If previous click didn't change the page, try a lower ranked element
                    idx = min(len(candidate_elements) - 1, limit)
                    candidate_ele = candidate_elements[idx]
                    msg = f"Since previously the URL has not changed, trying to click the Top-{min(len(candidate_elements), limit + 1)} login button instead: "
                else:
                    candidate_ele = candidate_elements[0]
                    msg = "Trying to click the Top-1 login button: "

                # Record layout BEFORE clicking
                tmp_path = "tmp.png"
                try:
                    driver.get_screenshot_as_file(tmp_path)
                    _, prev_screenshot_elements = self.layout_extractor(tmp_path)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

                # Perform transition
                element_text, current_url, *_ = page_transition(
                    driver=driver,
                    dom=candidate_ele,
                    save_html_path=save_html_path,
                    save_shot_path=save_shot_path
                )
                PhishLLMLogger.spit(msg + f'{element_text}', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)

                if current_url:
                    # Check whether page content changed
                    try:
                        driver.get_screenshot_as_file(tmp_path)
                        _, curr_screenshot_elements = self.layout_extractor(tmp_path)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                    ranking_model_refresh_page = has_page_content_changed(
                        curr_screenshot_elements=curr_screenshot_elements,
                        prev_screenshot_elements=prev_screenshot_elements
                    )
                    PhishLLMLogger.spit(f"Has the webpage changed? {ranking_model_refresh_page}",
                                        caller_prefix=PhishLLMLogger._caller_prefix, debug=True)

                    # Logo detection on the new page, then recurse (skip brand recog to preserve initial brand)
                    logo_box, reference_logo = self.detect_logo(save_shot_path)
                    return self.test(
                        current_url, reference_logo, logo_box,
                        save_shot_path, save_html_path, driver, limit + 1,
                        brand_recog_time, crp_prediction_time, clip_prediction_time,
                        ranking_model_refresh_page=ranking_model_refresh_page,
                        skip_brand_recognition=True,
                        company_domain=company_domain,
                        company_logo=company_logo
                    )
            else:
                msg = '[\U00002705] Benign'
                PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, screenshot_img

        # -- Default benign fallback --
        msg = '[\U00002705] Benign'
        PhishLLMLogger.spit(msg)
        return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, screenshot_img





