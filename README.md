# Multimodal Fever Prediction in Hospitalized Cancer Patients 


## Overview
This repository contains the codebase and analytical framework accompanying the journal paper: **“Development and Clinical Validation of a Multimodal AI Framework to Predict Persistent Fever During Antibiotic Therapy in Hospitalized Patients with Cancer.”**
The project presents a clinically oriented multimodal artificial intelligence (AI) framework designed to predict whether hospitalized cancer patients who remain febrile 24–48 hours after initiation of intravenous broad-spectrum antibiotics will continue to experience fever at the critical 48–72-hour antibiotic reassessment window.

By integrating structured electronic health record (EHR) data, longitudinal temperature forecasting, note-derived phenotypes, and CT-derived imaging features, this framework aims to **support antimicrobial stewardship, reduce unnecessary diagnostic escalation, and improve individualized clinical decision-making**.

---

## Key Features
- **Primary Prediction Task:** Persistent fever prediction at 48–72 hours after antibiotic initiation 
- **Multimodal Feature-Level Fusion:**  
  - Structured tabular clinical variables (demographics, labs, vitals, comorbidities)  
  - Time-series forecasting using Chronos-2  
  - Clinical note phenotyping via Qwen3-Next  
  - Thoracic CT feature extraction via Qwen3-VL
  - TabPFN as final prediction engine  
- **Validation Strategy:**  
  - Repeated nested 5-fold cross-validation 
  - Temporal holdout cohort  
  - External validation: Pooled regional hospitals and MIMIC-IV ICU  
  - Clinician validation study involving 15 clinicians
- **Inclusion Criteria:**  
  - Adult oncology/hematology patients 
  - Broad-spectrum IV antibiotic therapy  
  - Persistent fever 24–48h after treatment start 
  - Outcome: Fever persistence at 48–72h


---

## Repository Structure
```text
├── data-processing/            # Feature extraction and preprocessing pipelines
├── training-validation/        # Model training and evaluation
├── clinican-validation-study/  # Model training and evaluation
└── requirements.txt            # Python dependencies
```


> **Note:** Patient-level data are not included in this repository.

---

## Data Availability
Due to legal and ethical restrictions, **raw patient-level data cannot be shared publicly**.  
A **pseudo-anonymized dataset** may be made available for validation purposes upon reasonable request and subject to institutional approvals.

For data access inquiries, please contact the corresponding author.

---

## Reproducibility
This repository reflects the complete analytical pipeline used in the manuscript, including:
- Feature engineering  
- Model training and validation
- Clinician validation study resources

All analyses were performed in **Python 3.11**. Core dependencies include:

Exact package versions are specified in `requirements.txt`.

---

## Intended Use
This code is provided **for research and reproducibility purposes only**.  
The models are **not intended for direct clinical deployment** without prospective validation, local recalibration, and appropriate clinical governance.

---

## Citation
If you use this code, please cite the corresponding manuscript:

> Pucher G, et al. *Development and Clinical Validation of a Multimodal AI Framework to Predict Persistent Fever During Antibiotic Therapy in Hospitalized Patients with Cancer.* (in review)


---

## Contact
For questions regarding the code or the study:

**Christopher M. Sauer, MD MPH PhD**  
Laboratory for Clinical Research and Real-World Evidence  
Department of Hematology & Stem Cell Transplantation  
University Hospital Essen, Germany  
📧 christopher.sauer@uk-essen.de

**Gernot Pucher, MSc MSc**  
Laboratory for Clinical Research and Real-World Evidence  
Department of Hematology & Stem Cell Transplantation  
University Hospital Essen, Germany  
📧 gernot.pucher@uk-essen.de
