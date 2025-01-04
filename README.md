# FBDD
This repository is about Factual Bias Decomposition Dataset (FBDD).
This repository contains a script (`main.py`) that sends images and prompts to multiple Large Vision-Language Models (LVLM), including:
- **OpenAI GPT-4**
- **Anthropic Claude 3**
- **Google Generative AI (Gemini)**

The script collects the responses and stores the results in Excel files.

---

# Overview

1. **Script**  
   - `main.py` takes arguments to toggle specific prompts (counter-factual, context) and optionally filter the dataset by selected IDs.

2. **Data Assumptions**  
   - A CSV file (`../dataset/dataset_meta.csv`) that has columns such as `ID`, `Question`, and (optionally) `Counterfactual prompting`.
   - A folder (`../dataset/img_dataset/`) containing images named `<ID>.png`.
   - You may adjust `CSV_FILE_PATH` and `IMAGES_FOLDER_PATH` in the code to match your actual data paths.

3. **Output**  
   - Results are saved to a timestamped folder (e.g., `results/experiment_results_YYYYMMDD-HHMMSS`), with Excel files for each iteration.

---

# Requirements

- Python **3.7+** (recommended)
- A virtual environment (e.g., `venv`) or other environment manager
- API keys for:
  - **OpenAI** (`OPENAI_API_KEY`)
  - **Anthropic** (`ANTHROPIC_API_KEY`)
  - **Google Generative AI (Gemini)** (`GOOGLE_API_KEY`)
- Dependencies listed in `requirements.txt` (e.g., `pandas`, `requests`, `tqdm`, `Pillow`, etc.)

---

# Installation

1. **Clone or Download the Repository**
   ```
   git clone https://github.com/jtoyama4/FBDD.git
   cd FBDD
   ```

2. **(Optional) Create and Activate a Virtual Environment**

On macOS / Linux
```
python -m venv venv_name
source venv_name/bin/activate
```

On Windows
```
python -m venv venv_name
venv\Scripts\activate
```

3. **Install Dependencies**
```
pip install -r requirements.txt
```

# Usage

1. **API Keys**

This script expects the following environment variables to be set:
* OPENAI_API_KEY
* ANTHROPIC_API_KEY
* GOOGLE_API_KEY

You can export them in your shell or set them before running the script, for example:

```
export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
export ANTHROPIC_API_KEY='YOUR_ANTHROPIC_API_KEY'
export GOOGLE_API_KEY='YOUR_GOOGLE_API_KEY'
```

2. **Folder / File Structure**

By default, main.py looks for:
* CSV file at ../dataset/dataset_meta.csv
* Images in ../dataset/img_dataset/

Adjust the script’s constants (CSV_FILE_PATH, IMAGES_FOLDER_PATH) if your paths differ.

The folder structure is as follows:
```
FBDD
├── dataset
│   ├── img_dataset
│   └── dataset_meta.csv
├── scripts
│   ├── main.py
│   ├── results
│   └── selected_ids
└── requirements.txt
```


3. **Command-line Arguments**
* --cf:
Enable a counter-factual prompt.
* --ct:
Enable a context prompt.
* --selected_id:
Path to a text file listing specific IDs to analyze. One ID per line. If omitted, all IDs in the CSV are processed.

Example usage:
```
# Basic run (process all IDs)
python main.py

# Enable counter-factual prompt
python main.py --cf

# Enable context prompt
python main.py --ct

# Use a file listing specific IDs
python main.py --selected_id selected_ids.txt
```

4. **Output**
A new folder results/experiment_results_YYYYmmDD-HHMMSS will be created, containing the Excel files with results (e.g. result_YYYYmmDD-HHMMSS_0.xlsx, etc.).


