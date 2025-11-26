"""
Configuration module for chest X-ray classification experiments.
Contains all hyperparameters and experiment settings.
"""

# Simulation parameters
RANDOM_SEEDS = [42, 35, 20, 15]
DATASET_SIZES = [100, 250, 500, 1000]
DIFFUSION_FINETUNE_RATIO = 0.2
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
IMAGE_GEN_RATIOS = [0.5, 1, 2, 3, 5]

# Data paths
DATA_ROOT = "./new_data"
CLASSIFIER_DEV_CSV = "new_data/classifier_dev_full.csv"
TEST_CSV = "new_data/test.csv"

# Training parameters
TARGET_FEATURE = "Pleural Effusion"
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Model parameters
NUM_CLASSES = 2
FREEZE_LAYERS = True

# Training strategy
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_STEP_SIZE = 10
LR_SCHEDULER_GAMMA = 0.1

# Data loader parameters
NUM_WORKERS = 4
SHUFFLE_TRAIN = True

# Image parameters
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Augmentation parameters
RANDOM_HORIZONTAL_FLIP_P = 0.5
RANDOM_ROTATION_DEGREES = 10
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2

# Output directories
BASE_OUTPUT_DIR = "simulation_outputs"

# Random seeds for reproducibility
TORCH_SEED = 42
NUMPY_SEED = 42

# Image generation parameters
BASE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
GENERATION_OUTPUT_DIR = "./generated_data"
GENERATION_PROMPT = "posterior-anterior chest X-ray, diagnostic grayscale radiograph, showing right pleural effusion, balanced exposure, no text or labels"
GENERATION_NEGATIVE_PROMPT = "text, labels, watermark, logo, patient name, annotation, letters, numbers, gridlines, ultrasound, CT, MRI, PET, colored image, chest tube, ECG leads, external devices, border, frame, unrealistic anatomy, blur, overexposure, underexposure, distortion, duplicate organs"
GENERATION_NUM_INFERENCE_STEPS = 30
GENERATION_GUIDANCE_SCALE = 7.5
GENERATION_IMAGE_HEIGHT = 512
GENERATION_IMAGE_WIDTH = 512

# LoRA fine-tuning parameters
LORA_FINETUNE_CAPTION = "a chest x-ray showing pleural effusion"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_NUM_EPOCHS = 5
LORA_BATCH_SIZE = 8
LORA_LEARNING_RATE = 1e-4
LORA_GRADIENT_ACCUMULATION_STEPS = 4
LORA_MIXED_PRECISION = "fp16"
LORA_SAVE_STEPS = 500
LORA_OUTPUT_DIR = "./finetuned_models"
