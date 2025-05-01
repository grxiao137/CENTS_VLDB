# CENTS_VLDB

Prototype implementation of CENTS for VLDB

---

## Overview

This repository contains a working—but still under-development—prototype of **CENTS**, our cost-effective framework for LLM-based table understanding. 
We’re in the process of cleaning up the code and will publish detailed setup instructions and a fully polished release soon.

---

## Directory Layout

```text
.
├── data
│   ├── const.py               # constant values used throughout
│   └── topk-turl5652.pkl      # precomputed top-k outputs for SA
│   └── README.md              # pointers to benchmark files
├── src
│   ├── cdr
│   │   ├── base_scorer.py     # abstract scorer interface
│   │   ├── omhyb.py           # hybrid scoring implementation
│   │   ├── base_solver.py     # abstract solver interface
│   │   └── omcs.py            # budget-aware solver implementation
│   ├── tdr
│   │   ├── dataset.py         # data loader for fine-tuning, we only run this for SA
│   │   ├── model.py           # task reduction model architecture
│   │   ├── train.py           # fine-tuning script
│   │   └── gentopk_sa.py      # generate top-k
│   ├── connector
│   │   ├── base_connector.py  # abstract LLM connector interface
│   │   └── gpt_connector.py   # OpenAI GPT connector
│   ├── serializer
│   │   ├── base_ser.py        # abstract serializer interface
│   │   └── dfser.py           # per-column pandas serialization
│   ├── process-omhyb-omcs-cta.py  # preprocess CTA dataset
│   └── process-omhyb-omcs-re.py   # preprocess RE dataset
└── exp
    ├── requirements.txt       # pip libs used
    ├── sotab-cta-omhyb-omcs.py # SOTAB baseline for CTA
    ├── sotab-re-omhyb-omcs.py  # SOTAB baseline for RE
    ├── tabgptv2-cta.py        # TableGPTv2 CTA experiments
    ├── tabgptv2-re.py         # TableGPTv2 RE experiments
    └── turl-sa-tdr.py         # Schema Augmentation with TURL + Task Data Reduction

