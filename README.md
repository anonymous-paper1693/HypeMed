# Hypergraphs Meet Medicine: Advancing Medication Recommendations through Enhanced Relationship Modeling
This paper proposes a medication recommendation method that utilizes the hypergraph approach to capture three levels of intricate relationships in Electronic Health Records.

[//]: # (HypeMed is an innovative framework designed for medication recommendations by capturing intricate relationships within Electronic Health Records &#40;EHRs&#41;. Leveraging hypergraph contrastive learning, HypeMed considers patient history, medical entity interactions, and prescription patterns across different levels, resulting in highly accurate and balanced medication recommendations. It strikes a fine balance between precision and mitigating medication-related risks, thus enhancing patient safety and treatment efficacy.)

# Abstract
The field of AI-driven personalized medication recommendations has gained significant attention in recent years. However, existing models often overlook the nuanced relationships inherent in Electronic Health Records (EHRs), which can be categorized into three levels: visit-level, patient-level, and EHR-level. This limitation hinders their ability to provide precise and tailored medication recommendations. 
To address this issue, we present HypeMed: a Hypergraph-based Medication Recommendation Framework.
By utilizing the structural properties of hypergraphs, these three classes of relationships can be more comprehensively captured.
Comprising two core modules, HypeMed exploits the potential of the Medical Entity Hypergraph Contrastive Learning Module (MHCL) and the Relationship Enhanced Medication Prediction Module (REMP). MHCL focuses on modeling high-order group context relationships among medical entities. REMP leverages representations obtainbed from MHCL, strategically devising three channels to target visit-level, patient-level, and EHR-level relationships, thus amalgamating these insights to provide highly accurate medication predictions for each patient visit. 
% Rigorous experimentation on real-world MIMIC-III and MIMIC-IV datasets validates HypeMed's effectiveness, which not only demonstrates comparable recommendation accuracy to state-of-the-art methods but also excels in striking a finer balance between precision and medication risk control. 
Rigorous experimentation on real-world MIMIC-III and MIMIC-IV datasets validates the effectiveness of HypeMed. It not only surpasses state-of-the-art methods in recommendation accuracy but also excels in maintaining a relatively lower medication risk.
These results underscore the significant progress and potential of our proposed model, HypeMed, within the realm of medication recommendation.

## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Description
This project contains the necessary python scripts for HypeMed as well as the directions. 
Considering that unauthorized public access to the MIMIC-III and MIMIC-IV databases is prohibited, we do not provide the associated data. You can use our provided scripts to preprocess the raw data once you have obtained the relevant data.
## Requirements
```text
Python==3.8.36
Torch==1.13.1+cu116
NumPy==1.24.4
```

## Usage
We follow the preprossing procedures of [SafeDrug](https://github.com/ycq091044/SafeDrug/tree/archived).

Below is a guide on how to use the scripts. Before processing, you should put the necessary data in the `data` directory.

```bash
# Data Processing.
python data/processing.py # MIMIC-III
python data/processing_4.py #MIMIC-IV

# Contrastive Learning Pretraing (on MIMIC-III)
python HypeMed/HypeMedPretrain.py --pretrain --pretrain_epoch 300 --pretrain_lr 1e-3 --pretrian_weight_decay 1e-5 --mimic 3 --name example
# Training
python HypeMed/HypeMedPretrain.py --mimic 3 --name example
# Testing
python HypeMed/HypeMedPretrain.py --mimic 3 --Test --name example
# ablation study
python HypeMed/HypeMedPretrain.py --mimic 3 --channel_ablation mem --name example
```
You can explore all adjustable hyperparameters through the `HypeMed/config.py` file.
