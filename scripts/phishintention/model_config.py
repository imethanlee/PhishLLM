# Model configuration: load detector / siamese / OCR weights from configs/configs.yaml
import os
import yaml
from .modules.awl_detector import config_rcnn
from .modules.logo_matching import siamese_model_config, ocr_model_config

def get_absolute_path(relative_path):
    base_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base_path, relative_path))

def load_config(reload_targetlist=False):

    with open(os.path.join(os.path.dirname(__file__), 'configs/configs.yaml')) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    # Iterate through the configuration and update paths
    for section, settings in configs.items():
        for key, value in settings.items():
            if 'PATH' in key and isinstance(value, str):  # Check if the key indicates a path
                absolute_path = get_absolute_path(value)
                configs[section][key] = absolute_path

    AWL_MODEL = config_rcnn(
        cfg_path=configs['AWL_MODEL']['CFG_PATH'],
        weights_path=configs['AWL_MODEL']['WEIGHTS_PATH'],
        conf_threshold=configs['AWL_MODEL']['DETECT_THRE']
    )

    # siamese model
    SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']

    SIAMESE_MODEL = siamese_model_config(
        num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
        weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH']
    )

    OCR_MODEL = ocr_model_config(weights_path = configs['SIAMESE_MODEL']['OCR_WEIGHTS_PATH'])


    return AWL_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE