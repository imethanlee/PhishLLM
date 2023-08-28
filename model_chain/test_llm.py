import openai
import json
import torch
import clip
from phishintention.src.OCR_aided_siamese import pred_siamese_OCR
from model_chain.utils import *
from model_chain.web_utils import *
from model_chain.logger_utils import *
from paddleocr import PaddleOCR
import math
import os
from lxml import html
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
from xdriver.xutils.Logger import Logger
from field_study.draw_utils import draw_annotated_image_box
from typing import List, Tuple, Set, Dict, Optional, Union
from lavis.models import load_model_and_preprocess
from functools import lru_cache
import yaml
from tldextract import tldextract
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key2.txt').read()
os.environ['CURL_CA_BUNDLE'] = ''


class TestLLM():

    def __init__(self, phishintention_cls, param_dict, proxies=None):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proxies = proxies

        ## Ranking model
        self.clip_model, self.clip_preprocess = clip.load(param_dict['rank']['model_name'], device=self.device)
        if self.device == "cpu": # https://github.com/openai/CLIP/issues/57
            self.clip_model.float()
        state_dict = torch.load(param_dict['rank']['checkpoint_path'])
        self.clip_model.load_state_dict(state_dict)

        ## Image Captioning model
        self.caption_model, self.caption_preprocess, _ = load_model_and_preprocess(name=param_dict['logo_caption']['model_name'],
                                                                                   model_type=param_dict['logo_caption']['model_type'],
                                                                                   is_eval=True,
                                                                                   device=self.device)

        ## LLM
        self.LLM_model = param_dict["LLM_model"]
        self.brand_prompt = param_dict['brand_recog']['prompt_path']
        self.crp_prompt = param_dict['crp_pred']['prompt_path']
        self.phishintention_cls = phishintention_cls

        # OCR model
        try:
            self.default_ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=self.device == 'cuda')
        except MemoryError:
            self.default_ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu= False)
        self.ocr_language_list = param_dict['ocr']['supported_langs']

        # Load the Google API key and SEARCH_ENGINE_ID once during initialization
        self.API_KEY, self.SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]

        ## Load hyperparameters
        self.ocr_sure_thre, self.ocr_unsure_thre, self.ocr_local_best_window = param_dict['ocr']['sure_thre'], param_dict['ocr']['unsure_thre'], param_dict['ocr']['local_best_window']
        self.logo_expansion_ratio = param_dict['logo_caption']['expand_ratio']

        self.brand_recog_temperature, self.brand_recog_max_tokens = param_dict['brand_recog']['temperature'], param_dict['brand_recog']['max_tokens']
        self.brand_recog_sleep = param_dict['brand_recog']['sleep_time']
        self.brand_valid_k, self.brand_valid_siamese_thre = param_dict['brand_valid']['k'], param_dict['brand_valid']['siamese_thre']
        self.get_industry = param_dict['brand_recog']['ask_industry']
        self.industry_temperature, self.industry_max_tokens = param_dict['brand_recog']['industry']['temperature'], param_dict['brand_recog']['industry']['max_tokens']

        self.crp_temperature, self.crp_max_tokens = param_dict['crp_pred']['temperature'], param_dict['crp_pred']['max_tokens']
        self.crp_sleep = param_dict['crp_pred']['sleep_time']

        self.rank_max_uis, self.rank_batch_size = param_dict['rank']['max_uis_process'], param_dict['rank']['batch_size']
        self.rank_driver_sleep = param_dict['rank']['driver_sleep_time']
        self.rank_driver_script_timeout = param_dict['rank']['script_timeout']
        self.rank_driver_page_load_timeout = param_dict['rank']['page_load_timeout']
        self.interaction_limit = param_dict['rank']['depth_limit']

        # webhosting domains as blacklist
        self.webhosting_domains = [x.strip() for x in open('./datasets/hosting_blacklists.txt').readlines()]

    @lru_cache(maxsize=None) # Cache the results of tld extraction
    def extract_domain(self, domain):
        return tldextract.extract(domain)

    def detect_logo(self, save_shot_path: str) -> Tuple[Optional[List[float]], Optional[Image.Image]]:
        # Logo detection
        screenshot_img = Image.open(save_shot_path).convert("RGB")
        with open(save_shot_path, "rb") as image_file:
            screenshot_encoding = base64.b64encode(image_file.read())
        logo_boxes = self.phishintention_cls.return_all_bboxes4type(screenshot_encoding, 'logo')

        if (logo_boxes is not None) and len(logo_boxes)>0:
            logo_box = logo_boxes[0]  # get coordinate for logo
            x1, y1, x2, y2 = logo_box
            reference_logo = screenshot_img.crop((x1, y1, x2, y2))  # crop logo out
        else:
            reference_logo = None
            logo_box = None
        return logo_box, reference_logo

    def generate_webpage_ocr(self, shot_path: str, html_path: str) -> Tuple[List[str], List[List[float]], str]:
        '''
            Run OCR
            :param shot_path:
            :param html_path:
            :return:
        '''
        detected_text = ''
        ocr_text = []
        ocr_coord = []
        most_fit_lang = self.ocr_language_list[0]
        best_conf = 0
        most_fit_results = ''
        ocr = self.default_ocr_model

        for lang in self.ocr_language_list:
            if lang != 'en':
                try:
                    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, use_gpu=self.device == 'cuda')  # need to run only once to download and load model into memory
                except MemoryError:
                    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, use_gpu=False)  # need to run only once to download and load model into memory

            result = ocr.ocr(shot_path, cls=True)
            median_conf = np.median([x[-1][1] for x in result[0]])

            if math.isnan(median_conf): # no text is detected
                break
            if median_conf >= self.ocr_sure_thre: # confidence is so high
                most_fit_results = result
                break
            elif median_conf > best_conf and median_conf >= self.ocr_unsure_thre: # confidence is moderately high, need further checking
                best_conf = median_conf
                most_fit_lang = lang
                most_fit_results = result
            if best_conf > 0 and self.ocr_language_list.index(lang) - self.ocr_language_list.index(most_fit_lang) >= self.ocr_local_best_window:  # local best language
                break

        # OCR can return results
        if len(most_fit_results):
            most_fit_results = most_fit_results[0]
            ocr_text = [line[1][0] for line in most_fit_results]
            ocr_coord = [line[0][0] + line[0][2] for line in most_fit_results] # [x1, y1, x2, y2]
            detected_text = ' '.join(ocr_text)

        # if OCR does not work, use the raw HTML
        elif os.path.exists(html_path):
            with io.open(html_path, 'r', encoding='utf-8') as f:
                page = f.read()
            if len(page):
                dom_tree = html.fromstring(page, parser=html.HTMLParser(remove_comments=True))
                unwanted = dom_tree.xpath('//script|//style|//head')
                for u in unwanted:
                    u.drop_tree()
                html_text = ' '.join(dom_tree.itertext())
                html_text = re.sub(r"\s+", " ", html_text).split(' ')
                detected_text = ' '.join([x for x in html_text if x])

        return ocr_text, ocr_coord, detected_text

    def generate_logo_caption(self, img: Image.Image) -> str:

        raw_image = img.convert("RGB")
        image = self.caption_preprocess["eval"](raw_image).unsqueeze(0).to(self.device)
        result = self.caption_model.generate({"image": image})
        return ' '.join(result)

    def ask_industry(self, html_text):
        industry = ''
        if self.get_industry and len(html_text):
            prompt = question_template_industry(html_text)
            inference_done = False
            while not inference_done:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.LLM_model,
                        messages=prompt,
                        temperature=self.industry_temperature,
                        max_tokens=self.industry_max_tokens,  # we're only counting input tokens here, so let's not waste tokens on the output
                    )
                    inference_done = True
                except Exception as e:
                    Logger.spit('LLM Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, warning=True)
                    prompt[-1]['content'] = prompt[-1]['content'][:len(prompt[-1]['content']) // 2]
                    time.sleep(self.brand_recog_sleep)

            industry = ''.join([choice["message"]["content"] for choice in response['choices']])
            if len(industry) > 30:
                industry = ''

        return industry

    def brand_recognition_llm(self, reference_logo: Optional[Image.Image],
                              logo_box: Optional[List[float]],
                              ocr_text: List[str], ocr_coord: List[List[float]],
                              image_width: int, image_height: int) -> Tuple[Optional[str], Optional[Image.Image]]:
        '''
            Brand Recognition Model
            :param reference_logo:
            :param logo_box:
            :param ocr_text:
            :param ocr_coord:
            :param image_width:
            :param image_height:
            :return:
        '''
        company_domain, company_logo = None, None
        industry = ''
        if len(ocr_text) and self.get_industry:
            industry = self.ask_industry(' '.join(ocr_text))

        if reference_logo:
            # generation image caption for logo
            logo_caption = self.generate_logo_caption(reference_logo)
            logo_ocr = ''
            if len(ocr_coord):
                # get the OCR text description surrounding the logo
                expand_logo_box = expand_bbox(logo_box, image_width=image_width, image_height=image_height, expand_ratio=self.logo_expansion_ratio)
                overlap_areas = pairwise_intersect_area([expand_logo_box], ocr_coord)
                logo_ocr = np.array(ocr_text)[overlap_areas[0] > 0].tolist()
                logo_ocr = ' '.join(logo_ocr)
        else:
            logo_caption = ''
            logo_ocr = ' '.join(ocr_text)

        PhishLLMLogger.spit(f'Logo caption: {logo_caption}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
        PhishLLMLogger.spit(f'Logo OCR: {logo_ocr}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
        PhishLLMLogger.spit(f'Industry: {industry}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

        if len(logo_caption) > 0 or len(logo_ocr) > 0:
            # ask gpt to predict brand
            if self.get_industry:
                question = question_template_brand_industry(logo_caption, logo_ocr, industry)
            else:
                question = question_template_brand(logo_caption, logo_ocr)

            with open(self.brand_prompt, 'rb') as f:
                prompt = json.load(f)
            new_prompt = prompt
            new_prompt.append(question)

            inference_done = False
            while not inference_done:
                try:
                    start_time = time.time()
                    response = openai.ChatCompletion.create(
                        model=self.LLM_model,
                        messages=new_prompt,
                        temperature=self.brand_recog_temperature,
                        max_tokens=self.brand_recog_max_tokens,
                    )
                    inference_done = True
                except Exception as e:
                    Logger.spit('LLM Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, warning=True)
                    time.sleep(self.brand_recog_sleep) # retry

            answer = ''.join([choice["message"]["content"] for choice in response['choices']])
            PhishLLMLogger.spit(f"LLM prediction time: {time.time() - start_time}", debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
            PhishLLMLogger.spit(f'Detected brand: {answer}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

            # check the validity of the returned domain, i.e. liveness
            if len(answer) > 0 and is_valid_domain(answer):
                company_logo = reference_logo
                company_domain = answer
        else:
            PhishLLMLogger.spit('No logo description', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

        return company_domain, company_logo

    def brand_validation(self, company_domain: str, reference_logo: Image.Image) -> Tuple[bool, float, float]:
        ## Brand recognition model : result validation
        logo_cropping_time, logo_matching_time = 0, 0
        validation_success = False

        start_time = time.time()
        returned_urls = query2image(query=company_domain + ' logo',
                                    SEARCH_ENGINE_ID=self.SEARCH_ENGINE_ID, SEARCH_ENGINE_API=self.API_KEY,
                                    num=self.brand_valid_k,
                                    proxies=self.proxies)
        logos = get_images(returned_urls, proxies=self.proxies)
        logo_cropping_time = time.time() - start_time
        PhishLLMLogger.spit(f'Download logos from GImage time: {logo_cropping_time}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

        if len(logos) > 0:
            reference_logo_feat = pred_siamese_OCR(img=reference_logo,
                                                   model=self.phishintention_cls.SIAMESE_MODEL,
                                                   ocr_model=self.phishintention_cls.OCR_MODEL)
            start_time = time.time()
            sim_list = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(pred_siamese_OCR, logo,
                                           self.phishintention_cls.SIAMESE_MODEL,
                                           self.phishintention_cls.OCR_MODEL) for logo in logos]
                for future in futures:
                    logo_feat = future.result()
                    matched_sim = reference_logo_feat @ logo_feat
                    sim_list.append(matched_sim)

            if any([x > self.brand_valid_siamese_thre for x in sim_list]):
                validation_success = True

            logo_matching_time = time.time() - start_time
            PhishLLMLogger.spit(f'Logo matching time: {logo_matching_time}', debug=True,
                                caller_prefix=PhishLLMLogger._caller_prefix)

        if not validation_success:
            PhishLLMLogger.spit('Fails logo matching', debug=True,
                                caller_prefix=PhishLLMLogger._caller_prefix)
        return validation_success, logo_cropping_time, logo_matching_time

    def crp_prediction_llm(self, html_text: str) -> bool:
        '''
            Use LLM to classify credential-requiring page v.s. non-credential-requiring page
            :param html_text:
            :return:
        '''
        question = question_template_prediction(html_text)
        with open(self.crp_prompt, 'rb') as f:
            prompt = json.load(f)
        new_prompt = prompt
        new_prompt.append(question)

        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                response = openai.ChatCompletion.create(
                    model=self.LLM_model,
                    messages=new_prompt,
                    temperature=self.crp_temperature,
                    max_tokens=self.crp_max_tokens,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                inference_done = True
            except Exception as e:
                Logger.spit('LLM Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, warning=True)
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2] # maybe the prompt is too long, cut by half
                time.sleep(self.crp_sleep)

        answer = ''.join([choice["message"]["content"] for choice in response['choices']])
        PhishLLMLogger.spit(f'CRP prediction: {answer}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
        if 'A.' in answer:
            return True # CRP
        else:
            return False

    def ranking_model(self, url: str, driver: XDriver, ranking_model_refresh_page: bool) -> \
                                Tuple[Union[List, str], List[torch.Tensor], XDriver]:
        '''
            Use CLIP to rank the UI elements to find the most probable login button
            :param url:
            :param driver:
            :param ranking_model_refresh_page:
            :return:
        '''
        if ranking_model_refresh_page:
            try:
                driver.get(url)
                time.sleep(self.rank_driver_sleep)
            except Exception as e:
                PhishLLMLogger.spit('Exception {} when visiting the webpage'.format(e), caller_prefix=PhishLLMLogger._caller_prefix, warning=True)
                driver.quit()
                XDriver.set_headless()
                driver = XDriver.boot(chrome=True)
                driver.set_script_timeout(self.rank_driver_script_timeout)
                driver.set_page_load_timeout(self.rank_driver_page_load_timeout)
                time.sleep(self.rank_driver_sleep)
                return [], [], driver

        try:
            (btns, btns_dom),  \
                (links, links_dom), \
                (images, images_dom), \
                (others, others_dom) = driver.get_all_clickable_elements()
        except Exception as e:
            PhishLLMLogger.spit('Exception {} when getting all clickable UIs'.format(e), caller_prefix=PhishLLMLogger._caller_prefix, warning=True)
            return [], [], driver

        all_clickable = btns + links + images + others
        all_clickable_dom = btns_dom + links_dom + images_dom + others_dom

        # element screenshot
        candidate_uis = []
        candidate_uis_imgs = []
        for it in range(min(self.rank_max_uis, len(all_clickable))):
            try:
                driver.scroll_to_top()
                x1, y1, x2, y2 = driver.get_location(all_clickable[it])
            except Exception as e:
                PhishLLMLogger.spit('Exception {} when taking screenshot of UI id={}'.format(e, it), caller_prefix=PhishLLMLogger._caller_prefix, warning=True)
                continue

            if x2 - x1 <= 0 or y2 - y1 <= 0 or y2 >= driver.get_window_size()['height']//2: # invisible or at the bottom
                continue

            try:
                ele_screenshot_img = Image.open(io.BytesIO(base64.b64decode(all_clickable[it].screenshot_as_base64)))
                candidate_uis_imgs.append(self.clip_preprocess(ele_screenshot_img))
                candidate_uis.append(all_clickable_dom[it])
            except Exception as e:
                PhishLLMLogger.spit('Exception {} when when taking screenshot of UI id={}'.format(e, it), caller_prefix=PhishLLMLogger._caller_prefix, warning=True)
                continue

        # rank them
        if len(candidate_uis_imgs):
            PhishLLMLogger.spit(f'Find {len(candidate_uis_imgs)} candidate UIs', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            final_probs = torch.tensor([], device='cpu')
            batch_size = self.rank_batch_size
            texts = clip.tokenize(["not a login button", "a login button"]).to(self.device)

            for batch in range(math.ceil(len(candidate_uis)/batch_size)):
                chunked_images = candidate_uis_imgs[batch*batch_size : min(len(candidate_uis), (batch+1)*batch_size)]
                images = torch.stack(chunked_images).to(self.device)
                logits_per_image, _ = self.clip_model(images, texts)
                probs = logits_per_image.softmax(dim=-1)  # (N, C)
                final_probs = torch.cat([final_probs, probs.detach().cpu()], dim=0)
                del images

            conf = final_probs[torch.arange(final_probs.shape[0]), 1]  # take the confidence (N, 1)
            _, indices = torch.topk(conf, min(5, final_probs.shape[0]))  # top10 index
            candidate_uis_selected = [candidate_uis[ind] for ind in indices]
            candidate_imgs_selected = [candidate_uis_imgs[ind] for ind in indices]
            return candidate_uis_selected, candidate_imgs_selected, driver
        else:
            PhishLLMLogger.spit('No candidate login button to click', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            return [], [], driver


    def test(self, url: str, reference_logo: Optional[Image.Image],
             logo_box: Optional[List[float]],
             shot_path: str, html_path: str, driver: XDriver, limit: int=0,
             brand_recog_time: float=0, crp_prediction_time: float=0, crp_transition_time: float=0,
             ranking_model_refresh_page: bool=True,
             skip_brand_recognition: bool=False,
             brand_recognition_do_validation: bool=False,
             company_domain: Optional[str]=None, company_logo: Optional[Image.Image]=None,
             ):
        '''
            PhishLLM
            :param url:
            :param reference_logo:
            :param shot_path:
            :param html_path:
            :param driver:
            :param limit:
            :param brand_recog_time:
            :param crp_prediction_time:
            :param crp_transition_time:
            :param ranking_model_refresh_page:
            :param skip_brand_recognition:
            :param brand_recognition_do_validation:
            :param company_domain:
            :param company_logo:
            :return:
        '''

        ## Run OCR to extract text
        ocr_text, ocr_coord, detected_text = self.generate_webpage_ocr(shot_path, html_path)
        plotvis = Image.open(shot_path)
        image_width, image_height = plotvis.size

        ## Brand recognition model
        if not skip_brand_recognition:
            start_time = time.time()
            company_domain, company_logo = self.brand_recognition_llm(reference_logo, logo_box, ocr_text, ocr_coord, image_width, image_height)
            brand_recog_time += time.time() - start_time
            time.sleep(self.brand_recog_sleep) # fixme: allow the openai api to rest, not sure whether this help
        # check domain-brand inconsistency
        if company_domain:
            domain4pred = self.extract_domain(company_domain)
            domain4url = self.extract_domain(url)
            phish_condition = (domain4pred.domain != domain4url.domain) or (domain4pred.suffix != domain4url.suffix)
        else:
            phish_condition = False

        # Brand prediction results validation
        if phish_condition and (not skip_brand_recognition):
            if brand_recognition_do_validation and (reference_logo is not None): # we can check the validity by comparing the logo on the webpage with the logos for the predicted brand
                validation_success, logo_cropping_time, logo_matching_time = self.brand_validation(company_domain, reference_logo)
                brand_recog_time += logo_cropping_time
                brand_recog_time += logo_matching_time
                phish_condition = validation_success
            else: # alternatively, we can check the aliveness of the predicted brand
                validation_success = is_alive_domain(company_domain, self.proxies)
                phish_condition = validation_success

        if phish_condition:
            # CRP prediction model
            start_time = time.time()
            crp_cls = self.crp_prediction_llm(detected_text)
            crp_prediction_time += time.time() - start_time
            time.sleep(self.crp_sleep)

            if crp_cls: # CRP page is detected
                if company_domain in self.webhosting_domains:
                    PhishLLMLogger.spit('[\U00002705] Benign, since it is a brand providing cloud services')
                    return 'benign', 'None', brand_recog_time, crp_prediction_time, crp_transition_time, plotvis
                else:
                    plotvis = draw_annotated_image_box(plotvis, company_domain, logo_box)
                    PhishLLMLogger.spit(f'[\u26A0\uFE0F] Phishing discovered, phishing target is {company_domain}')
                    return 'phish', company_domain, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis
            else:
                # Not a CRP page => CRP transition
                if limit >= self.interaction_limit:  # reach interaction limit -> just return
                    PhishLLMLogger.spit('[\U00002705] Benign, reached interaction limit ...', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                    return 'benign', 'None', brand_recog_time, crp_prediction_time, crp_transition_time, plotvis

                # Ranking model
                start_time = time.time()
                candidate_elements, _, driver = self.ranking_model(url, driver, ranking_model_refresh_page)
                crp_transition_time += time.time() - start_time

                if len(candidate_elements):
                    save_html_path = re.sub("index[0-9]?.html", f"index{limit}.html", html_path)
                    save_shot_path = re.sub("shot[0-9]?.png", f"shot{limit}.png", shot_path)

                    if not ranking_model_refresh_page: # if previous click didnt refresh the page select the lower ranked element to click
                        PhishLLMLogger.spit(f"Since previously the URL has not changed, trying to click the Top-{min(len(candidate_elements), limit)} login button instead ...",
                                           caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                        candidate_ele = candidate_elements[min(len(candidate_elements), limit)]
                    else: # else, just click the top-1 element
                        PhishLLMLogger.spit("Trying to click the Top-1 login button ...",
                                            caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                        candidate_ele = candidate_elements[0]

                    current_url, *_ = page_transition(driver, candidate_ele, save_html_path, save_shot_path)
                    if current_url: # click success
                        ranking_model_refresh_page = current_url != url
                        PhishLLMLogger.spit(f"URL has changed? {ranking_model_refresh_page}", caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                        # logo detection on new webpage
                        logo_box, reference_logo = self.detect_logo(save_shot_path)
                        return self.test(current_url, reference_logo, logo_box,
                                         save_shot_path, save_html_path, driver, limit+1,
                                         brand_recog_time, crp_prediction_time, crp_transition_time,
                                         ranking_model_refresh_page=ranking_model_refresh_page,
                                         skip_brand_recognition=True, brand_recognition_do_validation=brand_recognition_do_validation,
                                         company_domain=company_domain, company_logo=company_logo)

        PhishLLMLogger.spit('[\U00002705] Benign')
        return 'benign', 'None', brand_recog_time, crp_prediction_time, crp_transition_time, plotvis



if __name__ == '__main__':

    # load hyperparameters
    with open('./param_dict.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls, param_dict=param_dict)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.proxy = "http://127.0.0.1:7890" # proxy
    web_func = WebUtil()

    sleep_time = 3; timeout_time = 60
    # XDriver.set_headless()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)  # fixme: you
    Logger.set_debug_on()

    all_links = [x.strip().split(',')[-2] for x in open('./datasets/Brand_Labelled_130323.csv').readlines()[1:]]

    root_folder = './datasets/dynapd'
    result = './datasets/dynapd_llm.txt'
    os.makedirs(root_folder, exist_ok=True)

    for ct, target in enumerate(all_links):
        # if ct <= 5470:
        #     continue
        hash = target.split('/')[3]
        target_folder = os.path.join(root_folder, hash)
        os.makedirs(target_folder, exist_ok=True)
        if os.path.exists(result) and hash in open(result).read():
            continue
        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')
        URL = f'http://127.0.0.5/{hash}'

        if os.path.exists(shot_path):
            try:
                driver.get(URL, click_popup=True, allow_redirections=False)
                time.sleep(2)
                Logger.spit(f'Target URL = {URL}', caller_prefix=XDriver._caller_prefix, debug=True)
                page_text = driver.get_page_text()
                error_free = web_func.page_error_checking(driver)
                if not error_free:
                    Logger.spit('Error page or White page', caller_prefix=XDriver._caller_prefix, debug=True)
                    continue

                if "Index of" in page_text:
                    # skip error URLs
                    error_free = web_func.page_interaction_checking(driver)
                    if not error_free:
                        Logger.spit('Error page or White page', caller_prefix=XDriver._caller_prefix,
                                    debug=True)
                        continue

            except Exception as e:
                Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                continue

            target = driver.current_url()
            logo_box, reference_logo = llm_cls.detect_logo(shot_path)
            pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, _ = llm_cls.test(target,
                                                                                                    reference_logo,
                                                                                                    logo_box,
                                                                                                    shot_path,
                                                                                                    html_path,
                                                                                                    driver,
                                                                                                    limit=3,
                                                                                                    brand_recognition_do_validation=False
                                                                                                    )
            with open(result, 'a+') as f:
                f.write(hash+'\t'+str(pred)+'\t'+str(brand)+'\t'+str(brand_recog_time)+'\t'+str(crp_prediction_time)+'\t'+str(crp_transition_time)+'\n')

    driver.quit()






