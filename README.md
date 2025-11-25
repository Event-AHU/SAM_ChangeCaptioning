# SAM_ChangeCaptioning
## Getting Started
### Installation

**1. Install requirements**

Install requirements using pip:

```bash
pip install -r requirements.txt
```

**2. Prepare dataset**

The data structure of LEVIR-CC is organized as follows:

```bash
├─/root/Data/LEVIR_CC/
        ├─LevirCCcaptions.json
        ├─images
             ├─train
             │  ├─A
             │  ├─B
             ├─val
             │  ├─A
             │  ├─B
             ├─test
             │  ├─A
             │  ├─B
```
Then extract the text files for each pair of image change descriptions in LEVIR-CC：

```bash
python preprocess_data.py
```

### Training

Ensure that the data preparation steps above are completed before proceeding to train the model：

```bash
python train.py
```
