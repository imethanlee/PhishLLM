import os
import argparse
import logging
from datetime import date

import yaml
from tqdm import tqdm
from selenium.common.exceptions import WebDriverException

from scripts.phishintention.model_config import load_config
from scripts.pipeline.phishvlm import PhishVLM
from scripts.utils.PhishIntentionWrapper import LogoDetector, LogoEncoder, LayoutDetector
from scripts.utils.web_utils import boot_driver, restart_driver
from scripts.utils.logger_utils import PhishLLMLogger

os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read().strip()
logging.getLogger("httpcore").setLevel(logging.WARNING)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the PhishVLM phishing-detection pipeline over a folder of websites.")
    parser.add_argument("--folder", default="./datasets/test_sites",
                        help="Folder of websites to test (each subfolder holds shot.png / info.txt / html.txt).")
    parser.add_argument("--config", default='./param_dict.yaml', help="Config .yaml path")
    args = parser.parse_args()

    PhishLLMLogger.set_debug_on()
    PhishLLMLogger.set_verbose(True)

    # load hyperparameters
    with open(args.config) as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    AWL_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE = load_config()
    logo_extractor = LogoDetector(AWL_MODEL)
    logo_encoder = LogoEncoder(SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE)
    layout_extractor = LayoutDetector(AWL_MODEL)

    # PhishVLM pipeline
    llm_cls = PhishVLM(param_dict=param_dict,
                       logo_encoder=logo_encoder,
                       logo_extractor=logo_extractor,
                       layout_extractor=layout_extractor)

    driver = boot_driver()

    day = date.today().strftime("%Y-%m-%d")
    result_txt = '{}_phishllm.txt'.format(day)

    if not os.path.exists(result_txt):
        with open(result_txt, "w+") as f:
            f.write("folder" + "\t")
            f.write("phish_prediction" + "\t")
            f.write("target_prediction" + "\t")  # write top1 prediction only
            f.write("brand_recog_time" + "\t")
            f.write("crp_prediction_time" + "\t")
            f.write("crp_transition_time" + "\n")


    for ct, folder in tqdm(enumerate(os.listdir(args.folder))):
        if folder in [x.split('\t')[0] for x in open(result_txt, encoding='ISO-8859-1').readlines()]:
            continue

        info_path = os.path.join(args.folder, folder, 'info.txt')
        html_path = os.path.join(args.folder, folder, 'html.txt')
        shot_path = os.path.join(args.folder, folder, 'shot.png')
        predict_path = os.path.join(args.folder, folder, 'predict.png')
        if not os.path.exists(shot_path):
            continue

        try:
            url = open(info_path, encoding='ISO-8859-1').read().strip()
            if not url:
                url = 'https://' + folder
        except FileNotFoundError:
            url = 'https://' + folder

        logo_box, reference_logo = llm_cls.detect_logo(shot_path)
        try:
            pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis = llm_cls.test(url=url,
                                                                                                            reference_logo=reference_logo,
                                                                                                            logo_box=logo_box,
                                                                                                            shot_path=shot_path,
                                                                                                            html_path=html_path,
                                                                                                            driver=driver,
                                                                                                            )
            driver.delete_all_cookies()
        except (WebDriverException) as e:
            print(f"Driver crashed or encountered an error: {e}. Restarting driver.")
            driver = restart_driver(driver)
            continue

        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                f.write(f"{folder}\t{pred}\t{brand}\t{brand_recog_time}\t{crp_prediction_time}\t{crp_transition_time}\n")
            if pred == 'phish':
                plotvis.save(predict_path)
        except UnicodeEncodeError:
            continue


    driver.quit()
