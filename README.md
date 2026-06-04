# PhishVLM

**Reference-based phishing detection without a predefined reference list — powered by Vision-Language Models.**

An extension of our USENIX Security 2024 work
*"Less Defined Knowledge and More True Alarms: Reference-based Phishing Detection without a Pre-defined Reference List."*

<p align="center">
  • <a href="https://www.usenix.org/conference/usenixsecurity24/presentation/liu-ruofan">Read our Paper</a> •
  • <a href="https://sites.google.com/view/phishllm">Visit our Website</a> •
  • <a href="https://sites.google.com/view/phishllm/experimental-setup-datasets?authuser=0#h.r0fy4h1fw7mq">Download our Datasets</a> •
  • <a href="#citation">Cite our Paper</a> •
</p>

---

## Table of Contents
- [Introduction](#introduction)
- [Framework](#framework)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
  - [Step 1: Install requirements](#step-1-install-requirements)
  - [Step 2: Install Google Chrome](#step-2-install-google-chrome)
  - [Step 3: Register two API keys](#step-3-register-two-api-keys)
- [Prepare a Dataset](#prepare-a-dataset)
- [Run PhishVLM](#run-phishvlm)
- [Understand the Output](#understand-the-output)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Introduction

Existing reference-based phishing detection:

- :x: Relies on a **predefined reference list**, which lacks comprehensiveness and incurs a high maintenance cost.
- :x: Does **not fully exploit the textual semantics** present on the webpage.

PhishVLM builds a reference-based phishing detector that is:

- ✅ **Free of a predefined reference list** — modern VLMs have encoded far more extensive brand–domain knowledge than any hand-curated list.
- ✅ **Chain-of-thought credential-taking prediction** — the credential-taking status is reasoned step-by-step directly from the screenshot.

## Framework

<img src="./figures/phishllm.png"/>

**Input:** a URL and its screenshot &nbsp;&nbsp;|&nbsp;&nbsp; **Output:** `Phish` / `Benign` and the phishing target brand.

- **Step 1 — Brand recognition.**
  Input: cropped logo screenshot. Output: the VLM's predicted brand domain.

- **Step 2 — Credential-Requiring-Page (CRP) classification.**
  Input: the webpage screenshot. The VLM chooses **A. Credential-taking page** or **B. Non-credential-taking page**.
  If `A`, go to Step 4; if `B`, go to Step 3.

- **Step 3 — CRP transition (only when Step 2 returns `B`).**
  Input: screenshots of all clickable UI elements. The most likely login UI is clicked, and the pipeline returns to **Step 1** with the updated webpage and URL (bounded by `rank.depth_limit`).

- **Step 4 — Decision.**
  A page is flagged as **phishing** when **all** of the following hold:
  1. the predicted brand's domain is **inconsistent** with the webpage's own domain; **and**
  2. **brand validation** passes — by default the on-page logo is matched against Google Image search results for the predicted brand (`brand_valid.activate: True`); if validation is disabled, the predicted brand domain is instead required to be **alive**; **and**
  3. the page is classified as a **credential-taking page** (Step 2 returns `A`).

  If the predicted brand is itself a **web-hosting / cloud provider** (see `datasets/hosting_blacklists.txt`), the page is treated as **benign**. Otherwise the page is reported as **benign**.

## Repository Structure

```text
PhishVLM/
├── param_dict.yaml                 # Pipeline hyper-parameters
├── requirements.txt
├── prompts/                        # VLM prompts (system + few-shot examples)
│   ├── brand_recog_prompt.json
│   ├── crp_pred_prompt.json
│   └── crp_trans_prompt.json
├── datasets/
│   ├── hosting_blacklists.txt      # Web-hosting / cloud-provider domains
│   ├── test_sites/                 # Bundled demo site (www.baidu.com)
│   ├── openai_key.txt              # (you create) OpenAI API key
│   └── google_api_key.txt          # (you create) Google Search key + engine id
├── figures/
└── scripts/
    ├── infer/
    │   └── run.py                  # Entry point (inference loop)
    ├── pipeline/
    │   └── phishvlm.py             # PhishVLM class — the 4-step pipeline
    ├── utils/                      # Web interaction, drawing, logging helpers
    │   ├── web_utils.py
    │   ├── draw_utils.py
    │   ├── logger_utils.py
    │   └── PhishIntentionWrapper.py
    └── phishintention/             # Logo detector / siamese / OCR backbones
        ├── model_config.py         # load_config(): builds the vision models
        ├── configs/                # *.yaml model configs
        ├── modules/                # detector + logo matching
        ├── ocr_lib/                # OCR-aided siamese encoder
        ├── utils/
        └── setup.sh                # Downloads pretrained weights
```

## Setup

> Tested on **Ubuntu** with an **NVIDIA GPU** and **CUDA 11**. A CPU-only run is
> possible but slow; the vision backbones fall back to CPU automatically.

### Step 1: Install requirements

A new conda environment named `phishvlm` is created in this step.

```bash
conda create -n phishvlm python=3.10 -y
conda activate phishvlm

# Python dependencies
pip install -r requirements.txt

# PyTorch (must match your CUDA version — example below is CUDA 11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# detectron2 (used by the logo / layout detector)
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

# Download the pretrained vision-model weights into scripts/phishintention/models/
cd scripts/phishintention
chmod +x setup.sh
./setup.sh
cd ../..
```

`setup.sh` downloads the layout/logo detector, the OCR-aided siamese encoder and
the supporting reference files via `gdown` and places them under
`scripts/phishintention/models/`. Re-running it is safe — existing files are skipped.

### Step 2: Install Google Chrome

PhishVLM drives a headless Chrome through Selenium. Install Chrome and a matching
driver (the driver is fetched automatically at runtime by `webdriver-manager`):

```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install -y ./google-chrome-stable_current_amd64.deb
```

### Step 3: Register two API keys

- 🔑 **OpenAI API key** — [tutorial](https://platform.openai.com/docs/quickstart).
  Paste the key into `./datasets/openai_key.txt`:

  ```bash
  echo "sk-your-openai-key" > ./datasets/openai_key.txt
  ```

- 🔑 **Google Programmable Search API key** — [tutorial](https://meta.discourse.org/t/google-search-for-discourse-ai-programmable-search-engine-and-custom-search-api/307107).
  Put the API key on the **first** line and the Search Engine ID on the **second** line of
  `./datasets/google_api_key.txt`:

  ```text
  [API_KEY]
  [SEARCH_ENGINE_ID]
  ```

> Both key files live under `datasets/` and are git-ignored, so your secrets are never committed.

## Prepare a Dataset

Organize the sites you want to test as one subfolder per website:

```text
testing_dir/
├── aaa.com/
│   ├── shot.png   # webpage screenshot
│   ├── info.txt   # webpage URL
│   └── html.txt   # webpage HTML source
├── bbb.com/
│   ├── shot.png
│   ├── info.txt
│   └── html.txt
└── ...
```

A ready-to-run example is provided in `datasets/test_sites/`.

## Run PhishVLM

Run from the **project root** as a module (this guarantees the `scripts` package is
importable):

```bash
conda activate phishvlm
python -m scripts.infer.run --folder ./datasets/test_sites
```

Optional arguments:

| Argument    | Default                  | Description                                   |
|-------------|--------------------------|-----------------------------------------------|
| `--folder`  | `./datasets/test_sites`  | Folder of websites to test.                   |
| `--config`  | `./param_dict.yaml`      | Pipeline hyper-parameter file.                |

## Understand the Output

- The console prints a live log, e.g.:

  <details><summary>Expand to see a sample log</summary>

  ```text
  [PhishLLMLogger][DEBUG] Folder ./datasets/field_study/2023-09-01/device-...remotewd.com
  [PhishLLMLogger][DEBUG] Time taken for LLM brand prediction: 0.97 Detected brand: sonicwall.com
  [PhishLLMLogger][DEBUG] Domain sonicwall.com is valid and alive
  [PhishLLMLogger][DEBUG] Time taken for LLM CRP classification: 2.92   CRP prediction: A. This is a credential-requiring page.
  [❗️] Phishing discovered, phishing target is sonicwall.com
  ```
  </details>

- A results file named `[today's date]_phishllm.txt` is written to the working directory
  (tab-separated). When a site is flagged as phishing, an annotated `predict.png` is saved
  inside that site's folder. Columns:

  | Column                | Meaning                                            |
  |-----------------------|----------------------------------------------------|
  | `folder`              | Website subfolder name.                            |
  | `phish_prediction`    | `phish` or `benign`.                               |
  | `target_prediction`   | Predicted target brand domain (e.g. `paypal.com`). |
  | `brand_recog_time`    | Time spent on brand recognition + validation (s).  |
  | `crp_prediction_time` | Time spent on CRP prediction (s).                  |
  | `crp_transition_time` | Time spent on CRP transition / ranking (s).        |

## Configuration

All pipeline knobs live in [`param_dict.yaml`](./param_dict.yaml), including:

- `VLM_model` — the OpenAI vision model used (default `gpt-4o-mini-2024-07-18`).
- `brand_recog`, `crp_pred`, `rank` — temperature, token limits and sleep/timeouts per step.
- `brand_valid` — whether to validate the predicted brand via logo matching, and the top-`k`
  / similarity threshold to use.
- `rank.depth_limit` — maximum number of CRP transitions (clicks) before giving up.

Model weights and detection thresholds are configured in
[`scripts/phishintention/configs/configs.yaml`](./scripts/phishintention/configs/configs.yaml).

## Troubleshooting

- **`ModuleNotFoundError: No module named 'scripts'`** — run the pipeline from the project
  root with the module form: `python -m scripts.infer.run` (not `python scripts/infer/run.py`).
- **`FileNotFoundError: openai_key.txt` / `google_api_key.txt`** — create the key files under
  `datasets/` as described in [Step 3](#step-3-register-two-api-keys).
- **Chrome / driver errors** — make sure Google Chrome is installed; `webdriver-manager`
  downloads the matching ChromeDriver on first run (requires network access).
- **CUDA / detectron2 build errors** — verify that your installed PyTorch CUDA build matches
  the CUDA toolkit on your machine before installing detectron2.

## Citation

```bibtex
@inproceedings{299838,
  author    = {Ruofan Liu and Yun Lin and Xiwen Teoh and Gongshen Liu and Zhiyong Huang and Jin Song Dong},
  title     = {Less Defined Knowledge and More True Alarms: Reference-based Phishing Detection without a Pre-defined Reference List},
  booktitle = {33rd USENIX Security Symposium (USENIX Security 24)},
  year      = {2024},
  isbn      = {978-1-939133-44-1},
  address   = {Philadelphia, PA},
  pages     = {523--540},
  url       = {https://www.usenix.org/conference/usenixsecurity24/presentation/liu-ruofan},
  publisher = {USENIX Association},
  month     = aug
}
```

If you have any issues running our code, please open a GitHub issue or email us:
liu.ruofan16@u.nus.edu, lin_yun@sjtu.edu.cn, dcsdjs@nus.edu.sg.
</content>
