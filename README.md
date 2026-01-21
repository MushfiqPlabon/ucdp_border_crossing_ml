# Hybrid Early Warning System (HEWS): Cross-Border Conflict Prediction

## Abstract
This repository implements a theoretical **Hybrid Early Warning System (HEWS)** designed to predict the likelihood of cross-border conflict events involving Bangladesh. Leveraging historical event-level data from the **Uppsala Conflict Data Program (UCDP)**, the system addresses the critical challenge of **extreme class imbalance** (N=8 border violations out of 693 total events). By integrating supervised probabilistic modeling with unsupervised anomaly detection, HEWS provides a dual-signal intelligence framework suitable for operational decision support.

## Methodology & Architecture

### 1. Data Pipeline
*   **Source:** UCDP Georeferenced Event Dataset (GED).
*   **Preprocessing:** Five-stage pipeline including automated keyword-based target engineering, temporal feature extraction (Monthly/Yearly), and spatial standardization.
*   **Imbalance Mitigation:** Implementation of **Manual Bootstrap Upsampling** within the cross-validation loop to stabilize minority class learning without introducing global data leakage.

### 2. Hybrid Model Logic
The system discards the traditional "single-best-classifier" approach in favor of a **Dual-Signal Hybrid Architecture**:
*   **Precision Signal (Supervised):** An `AdaBoost` ensemble utilizing decision stumps. This model is optimized for pattern recognition, identifying events that statistically align with historical border conflict characteristics.
*   **Sensitivity Signal (Unsupervised):** A `One-Class Support Vector Machine (SVM)` with a Radial Basis Function (RBF) kernel. Trained exclusively on internal conflict data, it serves as an outlier detector to flag novel or unusual geographic activity that supervised models might overlook.

### 3. Tactical Intelligence Reporting
The system translates machine learning metrics into a three-tier tactical alert framework:
*   **🔴 RED ALERT:** High correlation with historical border conflict patterns (Probability > 20%).
*   **🟡 YELLOW ALERT:** Anomaly detected. The event is statistically distinct from internal peace patterns, necessitating human investigation despite low supervised probability.
*   **🟢 GREEN:** Statistical alignment with routine internal conflict patterns.

## Technical Stack
*   **Environment:** Python 3.12.3 managed via `uv`.
*   **Modeling:** `scikit-learn` (Ensemble methods, SVM, Pipeline, GridSearchCV).
*   **Validation:** `RepeatedStratifiedKFold` (100 total iterations) to ensure statistical significance given the limited sample size.
*   **Interface:** `Gradio` for real-time tactical dashboarding.
*   **Analysis:** `ydata-profiling` for automated exploratory data analysis.

## Setup & Execution

### Prerequisites
Ensure the `uv` package manager is installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation
1. Initialize the environment and sync dependencies:
   ```bash
   uv sync
   ```
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

### Operational Use
*   **Training:** Execute the `ucdp_border_crossing_ml.ipynb` notebook to perform the full training, tuning, and validation cycle.
*   **Inference:** The notebook includes a Gradio interface (Task 10) that accepts Sector, Month, and Violence Type inputs to generate a Tactical Intelligence Report.
*   **Deployment:** The system is unified into a single `HybridTacticalModel` wrapper class, which can be exported via the included pickling logic for production integration.

## Academic & Professional Standards
This project adheres to rigorous software engineering and data science standards:
*   **Extrapolation Safety:** Utilizes robust scaling and non-linear kernels to handle future temporal inputs.
*   **Complexity Optimization:** Optimized upsampling logic ($O(N)$ complexity) for memory efficiency.
*   **Type Safety:** Implementation includes Python 3.12 type hinting for critical path logic.

---
*Developed as a theoretical operational planning tool for border security and alertness planning.*
