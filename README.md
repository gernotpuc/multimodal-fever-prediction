# Multimodal Fever Prediction in Hospitalized Cancer Patients 


## Overview
This repository contains the codebase and analytical framework accompanying the journal paper: **“Development and Clinical Validation of a Multimodal AI Framework to Predict Persistent Fever During Antibiotic Therapy in Hospitalized Patients with Cancer.”**
The project presents a clinically oriented multimodal artificial intelligence (AI) framework designed to predict whether hospitalized cancer patients who remain febrile 24–48 hours after initiation of intravenous broad-spectrum antibiotics will continue to experience fever at the critical 48–72-hour antibiotic reassessment window.

By integrating structured electronic health record (EHR) data, longitudinal temperature forecasting, note-derived phenotypes, and CT-derived imaging features, this framework aims to **support antimicrobial stewardship, reduce unnecessary diagnostic escalation, and improve individualized clinical decision-making**.


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



## Repository structure

```text
.
├── data-processing/                 # Preprocessing and feature-extraction utilities
├── training-validation/             # Model training, cross-validation, external validation and embeddings
├── clinican-validation-study/       # Web application used for the clinician vignette study
├── demo/                            # Synthetic runnable example and expected output
│   ├── example_input.csv
│   ├── run_demo.py
│   └── expected_output.txt
├── requirements-demo.txt            # Lightweight dependencies required for the runnable demo
├── LICENSE                          # MIT License for code authored in this repository
└── README.md
```


> **Note:** Patient-level data are not included in this repository.

## Quick-start demo

A lightweight runnable demonstration is provided so that readers can verify installation, input handling, model execution, and output generation without access to restricted clinical data or large foundation-model weights.

The demo uses the **released logistic-regression nested cross-validation pipeline** (`training-validation/lr_cv.py`) on a small synthetic encounter-level dataset. It deliberately uses the lightweight baseline rather than the full TabPFN/foundation-model stack so that it can be run on a standard CPU without model downloads or credentials.

The synthetic data contain **no real patient information** and the resulting performance estimates have **no clinical interpretation**. The demo is not intended to reproduce the numerical results in the manuscript.

### 1. Create an environment

The study analyses were developed using Python 3.11. For the demo:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements-demo.txt
```

### 2. Run the bundled example

From the repository root:

```bash
python demo/run_demo.py
```

The script reads:

```text
demo/example_input.csv
```

and creates its outputs under:

```text
demo/demo_output/
```

A successful run creates case-level predictions, pooled metrics, per-repeat metrics, threshold files, calibration outputs, error-analysis tables, and ROC/precision-recall/calibration plots. The main files are:

```text
demo/demo_output/nested_cv_predictions_case_level_pooled.csv
demo/demo_output/nested_cv_metrics_pooled.csv
demo/demo_output/nested_cv_calibration_pooled.csv
demo/demo_output/fold_thresholds.csv
demo/demo_output/roc_curve_pooled_repeats.png
demo/demo_output/precision_recall_curve_pooled_repeats.png
demo/demo_output/calibration_curve_pooled_repeats.png
```

Reference terminal output and the complete expected file list are provided in [`demo/expected_output.txt`](demo/expected_output.txt).

### Expected runtime

In a reference CPU-only test using 5 CPU cores of an AMD EPYC 7763 processor, the bundled demonstration completed in approximately **7 seconds**. On a typical modern desktop or laptop, it should generally complete in **well under one minute**. Runtime can vary with Python/package versions and hardware.

## Running the demo on your own data

`demo/run_demo.py` can also be used with a reader-supplied, **prepared encounter-level CSV**:

```bash
python demo/run_demo.py \
    --input /path/to/my_data.csv \
    --output-dir /path/to/demo_results \
    --target-col fever \
    --id-cols encounter_id
```

By default, all numeric or Boolean columns other than the target and ID columns are used as predictors. Explicit feature columns can instead be supplied:

```bash
python demo/run_demo.py \
    --input /path/to/my_data.csv \
    --output-dir /path/to/demo_results \
    --target-col fever \
    --id-cols encounter_id \
    --feature-cols age sex_male elixhauser_score time_lag_1_bt_max predictions_max
```

The target must be binary (`0`/`1`). The demo requires at least 30 rows because it runs stratified nested cross-validation. Missing predictor values should be handled before using the logistic-regression demo pipeline.

The bundled synthetic CSV illustrates the expected format. Its columns are:

| Column | Type | Meaning in the example |
|---|---|---|
| `encounter_id` | string | Synthetic encounter identifier |
| `age` | numeric | Age at admission |
| `sex_male` | binary | Encoded sex variable |
| `elixhauser_score` | numeric | Elixhauser comorbidity score |
| `length_stay` | numeric | Time from hospital admission to antibiotic initiation |
| `time_lag_1_bt_max` | numeric | Maximum body temperature 24–48 h after antibiotic initiation |
| `time_lag_2_bt_max` | numeric | Maximum body temperature 0–24 h after antibiotic initiation |
| `time_lag_1_crp_max` | numeric | Maximum CRP 24–48 h after antibiotic initiation |
| `time_lag_1_leua_max` | numeric | Maximum leukocyte count 24–48 h after antibiotic initiation |
| `predictions_max` | numeric | Maximum Chronos-2 forecasted temperature for 48–72 h |
| `fever` | binary | Example outcome: persistent fever during 48–72 h |

The complete engineered feature definitions used in the study are reported in Supplementary Table 24 of the accompanying manuscript.

## Running the released study analysis on prepared data

The training/validation scripts accept CSV or Parquet inputs and expose command-line options for data paths, target/ID columns, output locations, cross-validation settings, calibration, and thresholds.

### Tabularized multimodal TabPFN cross-validation

The primary tabularized multimodal model expects a prepared encounter-level table containing the structured and modality-derived features to be included in the analysis.

Example:

```bash
python training-validation/tabpfn_cv.py \
    --data-path /path/to/prepared_development_data.csv \
    --csv-sep , \
    --target-col fever \
    --id-cols encounter_id subject_reference \
    --output-dir results/tabpfn_cv \
    --outer-splits 5 \
    --inner-splits 5 \
    --n-repeats 20 \
    --bootstrap-iterations 2000 \
    --calibration-mode platt \
    --device auto
```

If `--feature-cols` is omitted, supported non-ID/non-target columns are inferred by the script. For strict reproducibility on another dataset, explicitly specifying the feature columns is recommended.

TabPFN must be installed separately for these analyses. Depending on the TabPFN release and execution environment, model access may also require a Hugging Face token supplied through the environment (for example, `HF_TOKEN`). No credentials should be stored in the repository.

### External validation of the TabPFN model

```bash
python training-validation/external_validate_tabpfn.py \
    --internal-data /path/to/prepared_development_data.csv \
    --external-data holdout=/path/to/holdout.csv external=/path/to/external.csv \
    --csv-sep , \
    --target-col fever \
    --id-cols encounter_id subject_reference \
    --output-dir results/tabpfn_external_validation \
    --calibration-mode platt \
    --threshold-oof-splits 5 \
    --bootstrap-iterations 2000 \
    --device auto
```

The external datasets must contain the feature columns used for model training. Thresholds and calibration are derived using internal data only by the released script.

### Logistic-regression baseline

The exact baseline exercised by the runnable demo can be run directly with more extensive settings:

```bash
python training-validation/lr_cv.py \
    --data-path /path/to/prepared_data.csv \
    --csv-sep , \
    --target-col fever \
    --id-cols encounter_id subject_reference \
    --output-dir results/logistic_regression_cv \
    --outer-splits 5 \
    --inner-splits 5 \
    --n-repeats 20 \
    --n-trials 50 \
    --bootstrap-iterations 2000
```

## Preparing raw data

The modelling scripts operate on encounter-level analysis tables, not directly on arbitrary raw EHR exports. The `data-processing/` directory contains the released preprocessing and feature-extraction utilities used to construct analysis-ready data.

For the structured clinical branch, `data-processing/run_preprocessing.py` expects a directory containing the measurement/intervention CSVs described in that script and produces encounter-level engineered features. Example:

```bash
python data-processing/run_preprocessing.py \
    --data-dir /path/to/preprocessing_inputs \
    --output-dir artifacts/preprocessed \
    --csv-sep ,
```

Additional modality-derived inputs are produced separately by the corresponding scripts, including Chronos-2 temperature features/embeddings, clinical-note features or embeddings, and imaging-derived features/embeddings. Some of these steps require external model weights, GPU resources, or access to an OpenAI-compatible local inference endpoint. See the docstring and command-line help of each script for its required inputs and environment variables.

Because local EHR schemas differ substantially between institutions, users applying the code to their own data are responsible for mapping local fields and timestamps to the study definitions before running the modelling scripts.



## Data availability

The institutional patient-level datasets used in the study cannot be deposited in this repository because they contain sensitive clinical information and are subject to data-protection and institutional restrictions.

A pseudonymized version of the minimum dataset required to reproduce the reported institutional analyses may be made available to qualified researchers affiliated with recognized academic or research institutions for non-commercial scientific research, subject to institutional review and approval and, where required, an appropriate data-use agreement. Please refer to the Data Availability statement in the manuscript for the current access procedure.

MIMIC-IV data are available separately through PhysioNet under credentialed access and the applicable data-use agreement. This repository does not redistribute MIMIC-IV data.



## Reproducibility notes

- The unit of analysis for model evaluation is the **hospital encounter/stay**.
- The primary target is persistent fever during **48–72 h after initiation of intravenous antibiotic therapy**.
- The public demo uses synthetic data and a reduced number of folds, tuning trials, and bootstrap iterations solely to keep runtime short.
- The manuscript analyses use the study-specific settings described in the Methods and Supplementary Information rather than the reduced demo settings.
- Random seeds are exposed in the command-line interfaces where applicable.
- Raw clinical data, pretrained foundation-model weights, and external model licences are not bundled with this repository.


## Intended Use
This code is provided **for research and reproducibility purposes only**.  
The models are **not intended for direct clinical deployment** without prospective validation, local recalibration, and appropriate clinical governance.


## Citation
If you use this code, please cite the accompanying manuscript:

> Pucher G, Kopp K, Deep A, et al. *Development and Clinical Validation of a Multimodal AI Framework to Predict Persistent Fever During Antibiotic Therapy in Hospitalized Patients with Cancer.* Nature Communications, forthcoming.
> 
## Software dependencies

The lightweight demo dependencies are specified in [`requirements-demo.txt`](requirements-demo.txt).

The full repository contains optional workflows with additional dependencies, including TabPFN, PyTorch, OpenAI-compatible inference clients, pydicom/Pillow, sentence-transformers, Chronos-2, Streamlit, and modality-specific model packages. These components may have their own hardware requirements, access requirements, and licences. Their use is governed by the respective upstream projects and model/data providers.

## License

Code authored in this repository is released under the **MIT License**. See [`LICENSE`](LICENSE).

Third-party libraries, pretrained models, datasets, model weights, and other external resources used by the software remain subject to their respective licences and terms of use. The MIT License in this repository does not relicense third-party components or clinical datasets.

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
