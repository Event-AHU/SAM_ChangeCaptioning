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
             ├─fine_features
             ├─semantic_features

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

### Testing 

Please run the following command：
```bash
python test.py
```

### Note: The complete execution code and weights will be uploaded later. The process includes the following steps:
Step 1: Use the modified SAM to extract motion-level features of the ROI.

Step 2: Use SAM combined with GroundingDINO to extract semantic-level features of the ROI.

Step 3: Feature processing.

Step 4: Graph extraction.

Step 5: Graph information encoding.
