# Melbourne Housing Price Prediction using Linear Regression

## Overview

This project implements Multivariate Linear Regression from scratch to predict Melbourne housing prices. The project compares three different implementations:

* **Part 1:** Pure Python implementation of Gradient Descent
* **Part 2:** NumPy-optimized implementation using vectorized operations
* **Part 3:** Scikit-learn implementation using `LinearRegression`

The objective is to study the effects of vectorization and optimized numerical libraries on training speed and model performance.

---

## Dataset

* **Dataset:** Melbourne Housing Dataset
* **Target Variable:** `Price`

### Features Used

* Rooms
* Bathroom
* Car
* YearBuilt
* Distance
* Lattitude
* Longtitude

---

## Project Structure

```text
melbourne-housing-linear-regression/
│
├── images/
├── report/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── preprocessing_numpy.py
│   ├── train_test_split.py
│   ├── train_test_split_numpy.py
│   ├── pure_python_regression.py
│   ├── numpy_regression.py
│   ├── sklearn_regression.py
│   └── evaluation.py
│
├── main_python.py
├── main_numpy.py
├── main_sklearn.py
├── compare_models.py
├── melb_data.csv
└── README.md
```

---

## Data Preprocessing

The following preprocessing steps were applied:

* Missing value imputation using column mean
* Winsorization for outlier treatment
* Mean Normalization
* Train-Test Split (80:20)

---

## Model Implementations

### Part 1 — Pure Python

Implemented from scratch using:

* Lists
* Loops
* Manual Gradient Descent

---

### Part 2 — NumPy

Optimized implementation using:

* NumPy arrays
* Vectorized matrix multiplication
* `np.dot`
* Broadcasting

---

### Part 3 — Scikit-learn

Implemented using:

```python
LinearRegression()
```

for comparison with the custom implementations.

---

## Evaluation Metrics

The models are evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Training Cost
* Training Time

---

## Visualizations

The project includes:

* Cost vs Iterations
* Cost vs Time
* Training Time Comparison
* Final Cost Comparison
* R² Score Comparison
* MAE Comparison
* RMSE Comparison

---

## Results

| Model        | Training Time | R² Score |  MAE  |  RMSE |
| ------------ | ------------: | -------: | ----: | ----: |
| Pure Python  |  12.8211      |  0.51524 |299519 |450236 |
| NumPy        |  0.18682      |  0.50895 |310370 |459989 |
| Scikit-learn |  0.01216      |  0.52662 |307352 |448051 |

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/kushagratripathi488-stack/melbourne-housing-linear-regression-with-pure-python
```

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

Run each implementation individually:

```bash
python main_python.py
python main_numpy.py
python main_sklearn.py
```

Generate comparison plots:

```bash
python compare_models.py
```

---

## Key Learning Outcomes

* Implemented Multivariate Linear Regression from scratch.
* Derived and implemented Gradient Descent manually.
* Optimized numerical computations using NumPy vectorization.
* Compared custom implementations with Scikit-learn.
* Evaluated models using standard regression metrics.
* Analyzed convergence behavior and computational performance.
