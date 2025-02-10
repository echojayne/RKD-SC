# Large Scale Model Enabled Semantic Communications Based on Robust Knowledge Distillation

This repository contains the implementation code for the paper:
**Large Scale Model Enabled Semantic Communications Based on Robust Knowledge Distillation**

This paper proposes a semantic communication system based on robust knowledge distillation (RKD-SC) for the lightweight application of large-scale semantic coding networks. We identify the problem of degraded robustness to noise in deep joint source and channel coding (JSCC) systems that are not trained jointly and propose a channel-aware autoencoder (CAA) based on the Transformer architecture to enhance the robustness of knowledge distillation-based semantic communication systems to channel noise.

## Overview
![](./images/method.png "Method")
Deep learning-based semantic communication systems have shown great promise in efficiently transmitting high-dimensional data. However, when the semantic coding networks are not jointly trained with channel models, their robustness against channel noise degrades. This work addresses the problem by introducing:

- **Robust Knowledge Distillation (RKD-SC):** A teacher-student framework that transfers knowledge from a high-capacity teacher to a lightweight student network.
- **Channel-Aware Autoencoder (CAA):** An autoencoder module leveraging the Transformer architecture, designed to mitigate the adverse effects of channel noise.
- **Comprehensive Evaluation:** Tools to simulate channel noise at different Signal-to-Noise Ratio (SNR) levels and evaluate system performance.

## Quick Start

Follow these steps to quickly get started with training and evaluating the proposed system:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```
2. **Install Dependencies:**

    Ensure you have Python 3.8 (or later) installed. Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
3.	**Prepare the Dataset:**

    The repository is configured to use standard datasets (e.g., CIFAR-10). Download and prepare the dataset as specified in the data/ directory or adjust the configuration in the provided config file.
4.	**Configure Hyperparameters:**

    Update the configuration file (config.yaml or similar) to set training parameters, learning rates, SNR levels, batch sizes, and other options.
5.	**Training:**

    To train the semantic communication system with robust knowledge distillation, run the notebook: `RKD-SC/experiments/distillation_student.ipynb`

## Repository Structure
A typical structure of the repository is as follows:

```json
├── README.md               # This file
├── requirements.txt        # List of required Python packages
├── models/                 # Contains model definitions (e.g., teacher, student, CAA)
├── data/                   # Data loading and preprocessing scripts
├── utils/                  # Utility functions 
└── experiments/            # Experimental notebooks and scripts for analysis
```
## Experimental Results

Our experiments demonstrate that the proposed RKD-SC system significantly improves robustness under noisy channel conditions compared to conventional JSCC methods. Detailed experimental results and analysis can be found in the paper and supplementary materials included with this repository.
![](./images/results.png "Method")
## Contact
For any questions or suggestions, please contact [dingkuiyuan@bupt.edu.cn]