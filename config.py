# config.py
# Experiment configuration (active scanning + selection strategies)

# ---------------------------
# Reproducibility
# ---------------------------
RANDOM_SEED = 20
N_RUNS = 2  # increase to 10–15 for final reporting

# ---------------------------
# Active scanning protocol
# ---------------------------
ROUNDS = 15
BATCH_SIZE = 25
INITIAL_SEED_SCANS = 20  # reduce for multiclass if first-hit coverage saturates too early

# ---------------------------
# Model settings
# ---------------------------
PCA_COMPONENTS = 8

# ---------------------------
# Strategy: ABC (Artificial Bee Colony)
# ---------------------------
ABC_COLONY_SIZE = 10
ABC_ITERS = 8
ABC_LIMIT = 6

# Fitness weights used by ABC/GA
# objective:  alpha*sum(p) + beta*sum(u) - gamma*sum(cost) - lambda*redundancy
ALPHA = 1.0
BETA = 0.15
GAMMA = 0.00
LAMBDA = 0.00  # set >0 only if you want feature redundancy control (adds overhead)

# ---------------------------
# Strategy: GA (Genetic Algorithm)
# ---------------------------
GA_POP_SIZE = 30
GA_GENERATIONS = 25
GA_ELITE_FRAC = 0.2
GA_TOURNAMENT_K = 3
GA_MUTATION_RATE = 0.3
GA_MUTATION_SWAPS = 2

# ---------------------------
# Strategy: BAMS-ABC (proposed)
# ---------------------------
# Batch split:
#   Bg = ceil(rho_g * B): greedy anchors (high p)
#   Bf = ceil(rho_f * B): forensic anchors (multiclass only)
#   Ba = B - Bg - Bf: ABC fill
BAMS_RHO_G = 0.40
BAMS_RHO_F = 0.40

# ---------------------------
# Dataset selection
# ---------------------------
USE_REAL_DATASET = True
DATASET_NAME = "ciciov2024"  # "ciciov2024" | "modbus2023" | "cicmalanal2017" | "dohbrw2020"
LABEL_MODE = "binary"        # "binary" or "multiclass"

# Sampling to keep runs fast (applies to loaders that support it)
SAMPLE_PER_CLASS = 1_000

RESULTS_DIR = "results"

# ---- Dataset paths ----
# CICIoV2024 (decimal CSV files)
CICIOV_DECIMAL_FILES = [
    "data/raw/ciciov2024/decimal/decimal_benign.csv",
    "data/raw/ciciov2024/decimal/decimal_DoS.csv",
    "data/raw/ciciov2024/decimal/decimal_spoofing-GAS.csv",
    "data/raw/ciciov2024/decimal/decimal_spoofing-RPM.csv",
    "data/raw/ciciov2024/decimal/decimal_spoofing-SPEED.csv",
    "data/raw/ciciov2024/decimal/decimal_spoofing-STEERING_WHEEL.csv",
]

# CICMalAnal2017 (processed multiclass CSV)
CICMALANAL2017_CSV = "data/processed/cicmalanal2017/cicmalanal2017_benign_scareware_adware.csv"

# CICModbus2023 (processed flows CSV)
MODBUS2023_FLOWS_CSV = "/Users/luiza/PycharmProjects/pca_rf_abc_scanner/datasets/CICModbus2023/processed/flows_labeled.csv"
MODBUS2023_SAMPLE_PER_CLASS = 50_000

# DoHBrw2020 (processed CSV)
DOHBRW2020_CSV = "data/processed/dohbrw2020/dohbrw2020_benign_malicious.csv"

# ---------------------------
# Caching (speed + reproducibility)
# ---------------------------
USE_SEED_CACHE = True
SEED_CACHE_DIR = "cache"

USE_MODEL_CACHE = True
MODEL_CACHE_DIR = "cache"

# ---------------------------
# Forensics settings (multiclass scoring + evaluation)
# ---------------------------
BENIGN_CLASS_ID = 0

# Per-sample multiclass scoring mode:
#   "off": p = 1 - P(benign)
#   "b3" : p = alpha*(1-P(benign)) + delta*sum_{unseen} P(class)
FORENSICS_MODE = "b3"
FORENSICS_ALPHA = 1.0
FORENSICS_DELTA = 2.0

# Confirmed coverage threshold (class is "confirmed" after >= M samples)
FORENSICS_CONFIRM_M = 2

# (Optional) batch-level forensics terms for ABC fitness
FORENSICS_KAPPA = 1.0  # expected new-class gain weight
FORENSICS_ETA = 0.2    # class-probability redundancy penalty

# ---------------------------
# Combined forensic score (FES) weights (reporting only)
# ---------------------------
FES_W_CCC = 0.5
FES_W_EBC = 0.3
FES_W_F1 = 0.2