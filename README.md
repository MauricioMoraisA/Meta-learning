# data imputation using meta-learning

This repository contains code and scripts used for experiments on data imputation.

## Repository Structure

```
meta/
│── imputation_Statistics/
│   ├── datasets/            # Datasets used in the experiments
│   ├── inputation.py        # Script for executing imputation
│   ├── separate_series.py   # Script for time series separation
│   ├── experiments.zip      # File containing the performed experiments
│── models/                  # Trained models
│── Moment/                  # Scripts related to statistical moment calculations
│── pix2pix/                 # Pix2Pix model implementation
│── create_class.py          # Class creation for modeling
│── environment.yml          # File for virtual environment configuration
│── inference.ipynb          # Analysis and visualization of results
│── meta.py                  # Main execution script
```

## Environment Setup

Before running the experiments, set up the environment:

```bash
conda env create -f environment.yml
conda activate environment_name
```

## Running the Experiments

### Option 1: Run the experiment from scratch

1. Navigate to the `imputation_Statistics/` directory and execute:

    ```bash
    cd meta/imputation_Statistics/
    python separate_series.py
    python inputation.py
    ```
    *Optional: Use Jupyter Notebook for interactive execution.*

2. Return to `meta/` and execute:

    ```bash
    cd Moment/
    python moment.py
    cd ../pix2pix/
    python pix2pix.py
    ```

3. Create the necessary classes:

    ```bash
    cd ../
    python create_class.py
    ```

4. Run the main script:

    ```bash
    python meta.py
    ```

5. To visualize results and figures, open and run `inference.ipynb` in Jupyter Notebook.

### Option 2: Reproduce the article results

To reproduce the exact experiments from the article:

1. Extract the `experiments.zip` file inside the `imputation_Statistics/` directory:
    ```bash
    unzip imputation_Statistics/experiments.zip -d imputation_Statistics/
    ```
2. Follow from step 5 of Option 1.

To check additional information, execute `work_preds`.

## Contact

If you have any questions, contact us at: mauricio.ma@discente.ufma.br

---

**Author:** Mauricio M. Almeida  
**License:** MIT
