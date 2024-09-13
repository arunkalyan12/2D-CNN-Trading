# Trading Strategy Project

This project aims to develop and test a trading strategy using a 2D Convolutional Neural Network (CNN) to analyze 1-minute OHLCV (Open, High, Low, Close, Volume) data from Coinbase. The goal is to predict profitable trading signals and make informed buy/sell decisions.

## Folder Structure

```
Trading_Strategy
├── Components
├── Config
│   ├── config.yaml
│   └── config_loader.py
├── Constants
├── Entity
├── Logging
├── Notebooks
├── Pipeline
├── Tests
│   └── test_config_loader.py
├── Utils
├── .gitignore
├── project_files.txt
└── README.md
└── requirements.txt
```

### **Directory Descriptions**

- **Components/**: Contains modules for specific tasks like data loading, preprocessing, model training, and evaluation.
- **Config/**: Configuration files including `config.yaml` for project settings and hyperparameters.
- **Constants/**: Files for storing constant values such as fixed hyperparameters and thresholds.
- **Entity/**: Data classes or schemas for defining the structure of important objects.
- **Logging/**: Custom logging configurations and scripts for tracking execution and metrics.
- **Pipeline/**: Scripts for end-to-end data processing, model training, and evaluation pipelines.
- **Tests/**: Unit tests to ensure the correctness of various components.
- **Utils/**: General utility functions and helper scripts.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **README.md**: This file.
- **requirements.txt**: List of project dependencies.

### **Configuration**

The configuration settings are stored in `Config/config.yaml`. This file includes paths, model parameters, and training configurations.

