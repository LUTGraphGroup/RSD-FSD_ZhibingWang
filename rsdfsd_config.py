import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(os.environ.get("RSD_FSD_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
CHINESE_DATA_ROOT = PROJECT_ROOT
# =============================== TRAINING ARGUMENTS ====================================
GROUP_BY_LENGTH = True
EPOCHS = 10
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_SCHEDULER_TYPE = "linear"
EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOGGING_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 2
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1_micro"
GREATER_IS_BETTER = True
OPTIM = "adamw_torch"
EARLY_STOPPING_PATIENCE = 5
MIN_TEXT_LENGTH = 512
MAX_TEXT_LENGTH = 6122
OVERLAP_WINDOW = 128
SEED = 888

# 0 means full training set. Keep full fkh_icd10 by default.
MAX_TRAIN_SAMPLES = 0

# =============================== EXPERIMENT SETTINGS ====================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CODE_TYPE = "desc"
DATA_TYPE = "fkh_icd10"
PRESISION_TYPR = "bf16"

# hfl/chinese-roberta-wwm-ext-large hidden size is 1024.
HIDDEN_SIZE = 1024
TRANSFORM_SIZE = 512
NUM_ATT_HEAD = 4
MODE = "train"

GRAPH_NUM = 1
BIAS_TYPE = "Y"
GRAPH_ATT = 512
GRAPH_FFN = 512

# =============================== CHINESE RoBERTa-WWM ====================================
PRETRAIN_MODEL = str(PROJECT_ROOT / "models" / "chinese-roberta-wwm-ext-large")
LOCAL_FILES_ONLY = True
DO_LOWER_CASE = False

# =============================== SSDC / HNCL ====================================
# fkh_icd10 currently has no static_anchors.pkl or hard_negative_bank.pkl by default.
# Keep these off for the first clean Chinese BERT run.
USE_SSDC = False
LAMBDA_SSDC = 0.10
SSDC_WARMUP_EPOCHS = 4

USE_U_SFC = False
USE_FG_HNCL = False
LAMBDA_HNCL = 0.005
HNCL_TOPK = 3
HNCL_MARGIN = 0.20
HNCL_STRATEGY = "top1"
HNCL_WARMUP_EPOCHS = 5
HNCL_COMPUTE_INTERVAL = 4

USFC_WEIGHT_TYPE = "distance"
USFC_MIN_WEIGHT = 0.05
USFC_MIX_ALPHA = 0.50

STRUCTURE_COVERAGE_RATIO = 0.70
TOPK_COMMON = 0
STATIC_ANCHOR_MODE = "full"
GATE_MODE = "three_stage"

USE_STAGE_GATE = True
STRUCTURE_GATE_VALUE = 1.0

# =============================== THRESHOLD CALIBRATION ====================================
USE_THRESHOLD_CALIBRATION = True
DEFAULT_THRESHOLD = 0.5
THRESHOLD_SEARCH_MIN = 0.05
THRESHOLD_SEARCH_MAX = 0.50
THRESHOLD_SEARCH_STEP = 0.01

# =============================== SRD-GTEP GRAPH SETTINGS ====================================
SRD_FEATURE_NAMES = [
    "p_r_given_c",
    "pmi",
    "support",
]
SRD_NUM_BINS = 16
SRD_TOPK_PER_TAIL = 16
SRD_SCORE_ALPHA = 1.00
SRD_SCORE_BETA = 1.00
SRD_SCORE_GAMMA = 0.75
SRD_SCORE_DELTA = 0.25
SRD_HUB_PENALTY_INIT = 0.05

# =============================== PRECISION ====================================
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16

# =============================== PATHS ====================================
START_MODEL_FROM_CHECKPOINT = ""
BASE_DIR = str(CHINESE_DATA_ROOT)
FILE_NAME = "seed{}_data{}_hflchineserobertawwmextlarge_bia{}_cod{}_graph{}_gratt{}".format(
    SEED, DATA_TYPE, BIAS_TYPE, CODE_TYPE, GRAPH_NUM, GRAPH_ATT
)
DATA_PATH = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE)
SAVE_METRICS_PATH = str(CHINESE_DATA_ROOT / "save" / DATA_TYPE / FILE_NAME / "metrics")
SAVE_PREDICTION_PATH = str(CHINESE_DATA_ROOT / "save" / DATA_TYPE / FILE_NAME / "predictions")
OUTPUT_DIR = str(CHINESE_DATA_ROOT / "save" / DATA_TYPE / FILE_NAME / "model")
THRESHOLD_FILE = str(Path(SAVE_METRICS_PATH) / "best_threshold.json")

STATIC_ANCHOR_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / "static_anchors.pkl")
HEAD_TAIL_MASK_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / "head_tail_mask.pkl")
ENHANCED_ICD_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / "icd_fkh_icd10_processed_v3.json")
PROCESSED_ICD_V3_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / "icd_fkh_icd10_processed_v3.json")
OFFICIAL_ICD_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / "icd_fkh_icd10.json")

ADJ_MATRIX_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / f"{DATA_TYPE}_adj_matrix_srd.pkl")
EDGE_FEATURE_INDICES_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / f"{DATA_TYPE}_edge_feature_indices_srd.pkl")
HEAD_DEGREE_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / f"{DATA_TYPE}_head_degree_srd.pkl")
GROUND_IND_TAIL_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / f"{DATA_TYPE}_ground_ind_tail.pkl")
GROUND_IND_HEAD_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / f"{DATA_TYPE}_ground_ind_head.pkl")
HARD_NEG_BANK_FILE = str(CHINESE_DATA_ROOT / "model_data" / DATA_TYPE / "hard_negative_bank.pkl")
