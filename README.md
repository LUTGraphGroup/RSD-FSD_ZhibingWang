# RSD-FSD

This repository provides the core research implementation of **RSD-FSD** for
multi-label clinical coding. Python identifiers use `RSDFSD` because hyphens
are not valid in class or module names.

## Scope

The public code includes:

- the RSD-FSD model and relation-aware graph encoder
- SRD graph and head-tail structure construction
- enhanced semantic anchors and hard-negative mining
- U-SFC, SSDC, and FG-HNCL training components
- dataset wrappers, metrics, and result export utilities
- fixed-prompt ICD label-text generation

The fixed-prompt generation entry is
`scripts/generate_enhanced_text.py`. Dataset-specific cleaning and conversion
code is intentionally not included.

The enhanced-text generator is the only refactored implementation. Other core
modules are organized from the original project with naming and import-path
adjustments for the public package layout.

## Layout

```text
rsdfsd/                   core package
scripts/                  text generation and result export
rsdfsd_config.py          local experiment settings
train_rsdfsd_chinese.py   training entry
```

## Data Benchmark

The dataset construction and train/development/test partitions follow the
**MIMIC-IV-ICD benchmark** [39], rather than a split defined by a particular
model:

- [MIMIC-IV-ICD9-Full data-processing code](https://github.com/thomasnguyen92/MIMIC-IV-ICD-data-processing/blob/master/notebooks/dataproc_mimic_IV_exploration_icd9.ipynb)
- [MIMIC-IV-ICD10-Full data-processing code](https://github.com/thomasnguyen92/MIMIC-IV-ICD-data-processing/blob/master/notebooks/dataproc_mimic_IV_exploration_icd10.ipynb)

Benchmark resources:

- [Official MIMIC-IV-ICD code repository](https://github.com/thomasnguyen92/MIMIC-IV-ICD-data-processing)
- [MIMIC-IV-ICD paper](https://arxiv.org/abs/2304.13998)
