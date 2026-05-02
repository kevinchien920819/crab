# Crab: Multimodal Deepfake and Emotion Detection

Crab is a research-oriented Python framework designed for **Deepfake Detection** and **Speech Emotion Recognition (SER)**. It utilizes state-of-the-art Self-Supervised Learning (SSL) models (e.g., Wav2Vec2, RoBERTa) and a custom architecture named **CrabNet** to achieve robust multimodal fusion of audio and text features.

---

## 🚀 Key Features

- **Multimodal Fusion:** Integrated audio and text processing using GRUs and Multi-head Attention.
- **Task Versatility:** Supports both synthetic speech detection (Deepfake) and emotion classification.
- **Dataset Support:** Pre-configured for major datasets including ASVspoof5, ASVspoof2021, MELD, IEMOCAP, and MSP-Podcast.
- **Experiment Tracking:** Native integration with [Weights & Biases (WandB)](https://wandb.ai/).
- **Modular Design:** Decoupled configuration, data pipeline, and model logic for easy extensibility.

---

## 📁 Project Structure

```text
├── configs/                # YAML configuration files
│   ├── deepfake/           # Deepfake detection experiment settings
│   └── emotion/            # SER (MELD, IEMOCAP, MSP) experiment settings
├── src/
│   ├── main.py             # Main entry point (orchestrates pipeline)
│   ├── config/             # Configuration schemas (Dataclasses)
│   │   ├── deepfake/       # Deepfake-specific config definitions
│   │   └── emotion/        # Emotion-specific config definitions
│   ├── controller/         # Training, testing, and evaluation logic
│   ├── data/               # Data loading, datasets, and samplers
│   ├── evaluation_metric/  # Metrics implementation (EER, a-DCF, etc.)
│   ├── model/              # Model architectures
│   │   ├── deepfake/       # Deepfake detection models
│   │   └── emotion/        # Emotion recognition models (CrabNet)
│   ├── tools/              # Third-party tools (e.g., LineBot integration)
│   └── utils/              # Logging, seeding, and shared utilities
├── preprocess/             # Data preprocessing scripts for each dataset
├── pyproject.toml          # Project dependencies (uv)
└── uv.lock                 # Dependency lock file
```

---

## 🛠️ Installation

This project uses `uv` for lightning-fast dependency management.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/crab.git
   cd crab
   ```

2. **Sync dependencies:**
   ```bash
   uv venv
   uv sync
   ```

---

## 💻 Usage

The system is controlled via the `src/main.py` entry point using YAML configurations stored in the `configs/` directory.

### Training & Evaluation

To run an experiment, provide the relative path to the configuration file (without the `.yaml` extension):

```bash
# Run Emotion Recognition on IEMOCAP
PYTHONPATH=src uv run -m src.main --config-name=emotion/IEMOCAP
```

### Configuration Parameters

Configurations are divided into several key sections:
- `general`: Execution flags (`train`, `eval`), device settings, and checkpoint paths.
- `model`: Architecture parameters like `ssl_model_str` (e.g., `WAV2VEC2_XLSR_300M`) and layer dimensions.
- `solver`: Optimizer settings, learning rate, and loss functions.
- `datasets`: Paths and subsets for training/testing.
- `wandb`: Experiment tracking toggles.

---

## 📊 Evaluation Metrics

Crab implements industry-standard metrics for objective performance assessment:
- **Deepfake Detection:** Equal Error Rate (EER), Act-DCF.
- **Emotion Recognition:** Unweighted Average Recall (UAR), Weighted Average Recall (WAR).

---

## 🔧 Development

### Adding a New Model
1. Define your architecture in `src/model/`.
2. Update the loader in `src/model/emotion/loader.py` or `src/model/deepfake/loader.py`.

### Adding a New Dataset
1. Implement the dataset logic in `src/data/dataset.py`.
2. Configure the directory paths in a new `.yaml` file in `configs/`.
