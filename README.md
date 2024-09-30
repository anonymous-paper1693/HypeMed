# HypeMed: Enhancing Medication Recommendations with Hypergraph-Based Patient Relationships
This paper proposes a medication recommendation method that utilizes the hypergraph approach to capture cross-patient relationships in Electronic Health Records.

![HypeMed](./assets/HypeMed.svg "Magic Gardens")
<!-- *The Architecture of HypeMed. The proposed HypeMed model consists of two main modules: the Medical Entity Hypergraph Contrastive Learning Module (MHCL) and the Relationship Enhanced Medication Prediction Module (REMP). MHCL is responsible for learning contextual representations of medical entities using hypergraph contrastive learning. REMP combines representations from multiple channels and utilizes a vector dot predictor to make medication recommendations.* -->

*The Architecture of HypeMed. HypeMed comprises two stages: the Medical Entity Relevance Representation Stage (MedRep) and the Similar Case Enhanced Medication Recommendation Stage (SimMR). MedRep focuses on modeling the global context-related entity relationships. SimMR models historical and similar case information to make recommendations.*

[//]: # (HypeMed is an innovative framework designed for medication recommendations by capturing intricate relationships within Electronic Health Records &#40;EHRs&#41;. Leveraging hypergraph contrastive learning, HypeMed considers patient history, medical entity interactions, and prescription patterns across different levels, resulting in highly accurate and balanced medication recommendations. It strikes a fine balance between precision and mitigating medication-related risks, thus enhancing patient safety and treatment efficacy.)

# Abstract
The AI-driven personalized medication recommendation field has gained significant attention in recent years. However, most existing algorithms focus primarily on modeling historical relationships within individual patient visit records, overlooking cross-patient relationships. This includes neglecting contextual connections beyond pairwise interactions between medical entities and the similarities between cases across different patients. This limitation hinders their ability to provide precise and tailored medication recommendations. To address this issue, this paper present **HypeMed**: a two-stage **Hype**rgraph-based **Med**ication recommendation framework. By leveraging the structural properties inherent in hypergraphs, it achieves a more comprehensive capture of the cross-patient relationships. HypeMed comprises two stages: the Medical Entity Relevance Representation Stage (MedRep) and the Similar Case Enhanced Medication Recommendation Stage (SimMR). MedRep focuses on modeling the global context-related relationships among multiple medical entities. SimMR establishes two channels to model the historical information within historical records and similar case information. By combining the two, highly accurate medication predictions for each patient visit can be provided. Rigorous experiments on real-world MIMIC-III and MIMIC-IV datasets validate the effectiveness of HypeMed, which surpasses current methods in recommendation accuracy and minimizes drug-drug interactions better than human experts. This demonstrates the significant potential of HypeMed in medication recommendation.

## Table of Contents
- [Description](#description)
- [Requirements](#requirements)
- [Usage](#usage)

## Description
This project contains the necessary python scripts for HypeMed as well as the directions. 
Considering that unauthorized public access to the MIMIC-III and MIMIC-IV databases is prohibited to distribute, we only provide the example data (first 100 entries of records of MIMIC-III and MIMIC-IV) in `data`. You can use our provided scripts to preprocess the raw data once you have obtained the relevant data.

## Requirements
```text
Python==3.8.36
Torch==1.13.1+cu116
NumPy==1.24.4
```

## Usage
<!-- We follow the preprossing procedures of [SafeDrug](https://github.com/ycq091044/SafeDrug/tree/archived). -->

Below is a guide on how to use the scripts. Before processing, please change the `'pathtomimic'` in `HypeMed.py` to the real path.

Please download **drug-DDI.csv**, which a large file, containing the drug DDI information. Please put it in `data`. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing

```bash
# Data Processing.
python data/processing.py # MIMIC-III
python data/processing_4.py #MIMIC-IV

# Contrastive Learning Pretraing (on MIMIC-III)
python HypeMed.py --pretrain --pretrain_epoch 3 --pretrain_lr 1e-3 --pretrian_weight_decay 1e-5 --mimic 3 --name example
# debug 
python HypeMed.py --debug --mimic 3 --pretrain_epoch 3 --name example
# Training
python HypeMed.py --mimic 3 --pretrain_epoch 3 --name example
# Testing
python HypeMed.py --mimic 3 --pretrain_epoch 3 --Test --name example
# ablation study
python HypeMed.py --mimic 3 --pretrain_epoch 3 --channel_ablation only_his --name example
```
You can explore all adjustable hyperparameters through the `config.py` file.
