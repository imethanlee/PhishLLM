from tldextract import tldextract
import openai
import time
import json
from PIL import Image
import io
import base64
import torch
import clip
import re
from xdriver.XDriver import XDriver
from phishintention.src.AWL_detector import find_element_type
from phishintention.src.OCR_aided_siamese import pred_siamese_OCR
from model_chain.utils import *
from paddleocr import PaddleOCR
import math
import os
from lxml import html
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
from tqdm import tqdm
import cv2
from model_chain.web_utils import WebUtil
from xdriver.xutils.Logger import Logger
import shutil
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

class TestLLM():

    def __init__(self, phishintention_cls):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.LLM_model = "gpt-3.5-turbo-16k"
        self.prediction_prompt = './selection_model/prompt3.json'
        self.brand_prompt = './brand_recognition/prompt.json'
        self.phishintention_cls = phishintention_cls
        self.language_list = ['en', 'ch', 'ru', 'japan', 'fa', 'ar', 'korean', 'vi', 'ms',
                             'fr', 'german', 'it', 'es', 'pt', 'uk', 'be', 'te',
                             'sa', 'ta', 'nl', 'tr', 'ga']

    def detect_text(self, shot_path, html_path):
        # ocr2text
        ocr_text = ''
        most_fit_lang = self.language_list[0]
        best_conf = 0
        most_fit_results = ''
        for lang in self.language_list:
            ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)  # need to run only once to download and load model into memory
            result = ocr.ocr(shot_path, cls=True)
            median_conf = np.median([x[-1][1] for x in result[0]])
            if math.isnan(median_conf):
                break
            if median_conf > best_conf and median_conf >= 0.9:
                best_conf = median_conf
                most_fit_lang = lang
                most_fit_results = result
            if median_conf >= 0.98:
                most_fit_results = result
                break
            if best_conf > 0:
                if self.language_list.index(lang) - self.language_list.index(most_fit_lang) >= 2:  # local best
                    break
        if len(most_fit_results):
            most_fit_results = most_fit_results[0]
            ocr_text = ' '.join([line[1][0] for line in most_fit_results])

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
                ocr_text = ' '.join([x for x in html_text if x])

        return ocr_text

    def url2logo(self, driver, URL):
        '''
            Get page's logo from an URL
            Args:
                URL:
                url4logo: the URL is a logo image already or not
        '''
        try:
            driver.get(URL, allow_redirections=False)
            time.sleep(3)  # fixme: must allow some loading time here
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            return None, str(e)

        # the URL is for a webpage not the logo image
        try:
            screenshot_encoding = driver.get_screenshot_encoding()
            screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
            screenshot_img = screenshot_img.convert("RGB")
            screenshot_img_arr = np.asarray(screenshot_img)
            screenshot_img_arr = np.flip(screenshot_img_arr, -1)  # RGB2BGR
            pred = self.phishintention_cls.AWL_MODEL(screenshot_img_arr)
            pred_i = pred["instances"].to('cpu')
            pred_classes = pred_i.pred_classes.detach().cpu()  # Boxes types
            pred_boxes = pred_i.pred_boxes.tensor.detach().cpu()  # Boxes coords

            if pred_boxes is None or len(pred_boxes) == 0:
                all_logos_coords = None
            else:
                all_logos_coords, _ = find_element_type(pred_boxes=pred_boxes,
                                                       pred_classes=pred_classes,
                                                       bbox_type='logo')
            if all_logos_coords is None:
                return None, 'no_logo_prediction'
            else:
                logo_coord = all_logos_coords[0]
                logo = screenshot_img.crop((int(logo_coord[0]), int(logo_coord[1]), int(logo_coord[2]), int(logo_coord[3])))
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            return None, str(e)

        return logo, 'success'

    def click_and_save(self, driver, URL, dom, save_html_path, save_shot_path):
        try:
            driver.get(URL, allow_redirections=False)
            time.sleep(3)  # fixme: must allow some loading time here
            element = driver.find_elements_by_xpath(dom)
            if element:
                driver.move_to_element(element[0])
                driver.click(element[0])
                time.sleep(2)  # fixme: must allow some loading time here
            current_url = driver.current_url()
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            return None, None, None

        try:
            driver.save_screenshot(save_shot_path)
            with open(save_html_path, "w", encoding='utf-8') as f:
                f.write(driver.page_source())
            return current_url, save_html_path, save_shot_path
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            return None, None, None

    def knowledge_cleansing(self, reference_logo, logo_list,
                            reference_domain, domain_list,
                            reference_tld, tld_list,
                            ):
        '''
            Knowledge cleansing with Logo matcher and Domain matcher
            Args:
                reference_logo: logo on the testing website as reference
                logo_list: list of logos to check, logos_status
                reference_domain: domain for the testing website
                reference_tld: top-level domain for the testing website
                domain_list: list of domains to check
                ts: logo matching threshold
                strict: strict domain matching or not
            Returns:
                domain_matched_indices
                logo_matched_indices
        '''

        domain_matched_indices = []
        for ct in range(len(domain_list)):
            if domain_list[ct] == reference_domain and tld_list[ct] == reference_tld:
                domain_matched_indices.append(ct)

        logo_matched_indices = []
        if reference_logo is not None:
            reference_logo_feat = pred_siamese_OCR(img=reference_logo,
                                                 model=self.phishintention_cls.SIAMESE_MODEL,
                                                 ocr_model=self.phishintention_cls.OCR_MODEL)
            for ct in range(len(logo_list)):
                if logo_list[ct] is not None:
                    logo_feat = pred_siamese_OCR(img=logo_list[ct],
                                                 model=self.phishintention_cls.SIAMESE_MODEL,
                                                 ocr_model=self.phishintention_cls.OCR_MODEL)
                    if reference_logo_feat @ logo_feat >= self.phishintention_cls.SIAMESE_THRE_RELAX:  # logo similarity exceeds a threshold
                        logo_matched_indices.append(ct)

        return domain_matched_indices, logo_matched_indices

    def brand_recognition_llm(self, url,
                              reference_logo, html_text,
                              driver,
                              do_validation=False
                              ):

        company_domain, company_logo = None, None
        q_domain, q_tld = tldextract.extract(url).domain, tldextract.extract(url).suffix
        question = question_template_brand(html_text)

        with open(self.brand_prompt, 'rb') as f:
            prompt = json.load(f)
        new_prompt = prompt
        new_prompt.append(question)

        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                start_time = time.time()
                response = openai.ChatCompletion.create(
                    model=self.LLM_model,
                    messages=new_prompt,
                    temperature=0,
                    max_tokens=50,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                inference_done = True
            except Exception as e:
                Logger.spit('LLM Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
                time.sleep(10)
        answer = ''.join([choice["message"]["content"] for choice in response['choices']])
        print('LLM prediction time:', time.time() - start_time)

        if len(answer) > 0 and len(answer) < 30: # has prediction
            if do_validation:
                # get the logos from knowledge sites
                start_time = time.time()
                logo, status = self.url2logo(driver=driver, URL='https://'+answer)
                print('Crop the logo time:', time.time() - start_time)

                if logo:
                    # Domain matching OR Logo matching
                    start_time = time.time()
                    logo_domains = [tldextract.extract(answer).domain]
                    logo_tlds = [tldextract.extract(answer).suffix]

                    domain_matched_indices, logo_matched_indices = self.knowledge_cleansing(reference_logo=reference_logo,
                                                                                            logo_list=[logo],
                                                                                            reference_domain=q_domain,
                                                                                            domain_list=logo_domains,
                                                                                            reference_tld=q_tld,
                                                                                            tld_list=logo_tlds)

                    domain_or_logo_matched_indices = list(set(domain_matched_indices + logo_matched_indices))

                    if len(domain_or_logo_matched_indices) > 0:
                        company_logo = logo
                        company_domain = answer
                    print('Logo matching or domain matching time:', time.time()-start_time)
            else:
                company_logo = reference_logo
                company_domain = answer

        return company_domain, company_logo

    def crp_prediction_llm(self, html_text):

        question = question_template_prediction(html_text)
        with open(self.prediction_prompt, 'rb') as f:
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
                    temperature=0,
                    max_tokens=100,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                inference_done = True
            except Exception as e:
                Logger.spit('LLM Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                # new_prompt = new_prompt[:65540]
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
                time.sleep(10)

        answer = ''.join([choice["message"]["content"] for choice in response['choices']])
        if 'A.' in answer:
            return 0 # CRP
        else:
            return 1

    def ranking_model(self, url, driver):

        try:
            driver.get(url)
            time.sleep(5)
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            driver.quit()
            XDriver.set_headless()
            driver = XDriver.boot(chrome=True)
            driver.set_script_timeout(30)
            driver.set_page_load_timeout(60)
            time.sleep(3)
            return [], []

        try:
            (btns, btns_dom),  \
                (links, links_dom), \
                (images, images_dom), \
                (others, others_dom) = driver.get_all_clickable_elements()
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            return [], []

        all_clickable = btns + links + images + others
        all_clickable_dom = btns_dom + links_dom + images_dom + others_dom

        # element screenshot
        candidate_uis = []
        candidate_uis_imgs = []
        for it in range(min(300, len(all_clickable))):
            try:
                driver.scroll_to_top()
                x1, y1, x2, y2 = driver.get_location(all_clickable[it])
            except Exception as e:
                Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                continue

            if x2 - x1 <= 0 or y2 - y1 <= 0 or y2 >= driver.get_window_size()['height']//2: # invisible or at the bottom
                continue

            try:
                ele_screenshot_img = Image.open(io.BytesIO(base64.b64decode(all_clickable[it].screenshot_as_base64)))
                candidate_uis_imgs.append(self.clip_preprocess(ele_screenshot_img))
                candidate_uis.append(all_clickable_dom[it])
            except Exception as e:
                Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                continue

        # rank them
        if len(candidate_uis_imgs):
            images = torch.stack(candidate_uis_imgs).to(self.device)
            texts = clip.tokenize(["not a login button", "a login button"]).to(self.device)

            logits_per_image, logits_per_text = self.clip_model(images, texts)
            probs = logits_per_image.softmax(dim=-1)  # (N, C)
            conf = probs[torch.arange(probs.shape[0]), 1]  # take the confidence (N, 1)
            _, ind = torch.topk(conf, 1)  # top1 index

            return candidate_uis[ind], candidate_uis_imgs[ind]
        else:
            return [], []


    def test(self, url, shot_path, html_path, driver, limit=3,
             brand_recog_time=0, crp_prediction_time=0, crp_transition_time=0):


        html_text = self.detect_text(shot_path, html_path)
        _, reference_logo = self.phishintention_cls.predict_n_save_logo(shot_path)
        start_time = time.time()
        company_domain, company_logo = self.brand_recognition_llm(url, reference_logo, html_text, driver)
        brand_recog_time += time.time() - start_time

        if company_domain:
            start_time = time.time()
            crp_cls = self.crp_prediction_llm(html_text)
            crp_prediction_time += time.time() - start_time

            if crp_cls == 0: # CRP
                if company_domain != tldextract.extract(url).domain+'.'+tldextract.extract(url).suffix:
                    return 'phish', company_domain, brand_recog_time, crp_prediction_time, crp_transition_time
            elif limit > 0: # CRP transition
                start_time = time.time()
                candidate_dom, candidate_img = self.ranking_model(url, driver)
                crp_transition_time += time.time() - start_time

                if len(candidate_dom):
                    save_html_path = re.sub("index[0-9]?.html", f"index{limit}.html", html_path)
                    save_shot_path = re.sub("shot[0-9]?.png", f"shot{limit}.png", shot_path)
                    current_url, *_ = self.click_and_save(driver, url, candidate_dom, save_html_path, save_shot_path)
                    if current_url: # click success
                        return self.test(current_url, save_shot_path, save_html_path, driver, limit-1,
                                         brand_recog_time, crp_prediction_time, crp_transition_time)

        return 'benign', 'None', brand_recog_time, crp_prediction_time, crp_transition_time



if __name__ == '__main__':

    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy
    web_func = WebUtil()

    sleep_time = 3; timeout_time = 60
    XDriver.set_headless()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)  # fixme: you
    Logger.set_debug_on()

    driver.get('http://phishing.localhost')
    time.sleep(3)
    driver.save_screenshot('./debug.png')
    all_links = [x.strip().split(',')[-2] for x in open('./datasets/Brand_Labelled_130323.csv').readlines()[1:]]

    root_folder = './datasets/dynapd'
    result = './datasets/dynapd_wo_validation.txt'
    os.makedirs(root_folder, exist_ok=True)

    for target in all_links:
        url = target
        print(target)
        hash = target.split('/')[3]
        target_folder = os.path.join(root_folder, hash)
        os.makedirs(target_folder, exist_ok=True)
        if os.path.exists(result) and hash in open(result).read():
            continue

        try:
            driver.get(target, click_popup=True, allow_redirections=False)
            time.sleep(5)
            Logger.spit(f'Target URL = {target}', caller_prefix=XDriver._caller_prefix, debug=True)
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            shutil.rmtree(target_folder)
            continue

        white_page = web_func.page_white_screen(driver, ts=0.9)
        # if (error_free == False) and (white_page == True):
        if (white_page == True):
            time.sleep(5)
            try:
                # skip error URLs
                error_free = web_func.page_interaction_checking(driver)
                white_page = web_func.page_white_screen(driver, ts=0.9)
                if (error_free == False) and (white_page == True):
                    continue
                target = driver.current_url()
            except Exception as e:
                Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                shutil.rmtree(target_folder)
                continue

        # skip URL which redirects to benign
        if tldextract.extract(target).domain != '127.0.0.5':
            shutil.rmtree(target_folder)
            continue
        if driver.current_url().endswith('https/') or driver.current_url().endswith('genWeb/'):
            shutil.rmtree(target_folder)
            continue

        try:
            shot_path = os.path.join(target_folder, 'shot.png')
            html_path = os.path.join(target_folder, 'index.html')
            # take screenshots
            screenshot_encoding = driver.get_screenshot_encoding()
            screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
            screenshot_img.save(shot_path)
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, warning=True)
            shutil.rmtree(target_folder)
            continue

        try:
            # record HTML
            with open(html_path, 'w+', encoding='utf-8') as f:
                f.write(driver.page_source())
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            pass

        if os.path.exists(shot_path):
            pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time = llm_cls.test(url, shot_path, html_path, driver)
            with open(result, 'a+') as f:
                f.write(hash+'\t'+pred+'\t'+brand+'\t'+str(brand_recog_time)+'\t'+str(crp_prediction_time)+'\t'+str(crp_transition_time)+'\n')

    driver.quit()






