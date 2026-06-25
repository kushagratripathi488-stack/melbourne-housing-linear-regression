# Melbourne Housing Price Prediction using Multivariable Linear Regression

## Project Overview

This project implements a Multivariable Linear Regression model from scratch using pure Python and Gradient Descent. The objective is to predict housing prices from the Melbourne Housing Dataset while understanding the mathematical foundations of machine learning without relying on external machine learning libraries.

The project focuses on:

* Data preprocessing
* Feature normalization
* Gradient Descent optimization
* Cost function convergence
* Model evaluation using regression metrics

---

## Dataset

Dataset: Melbourne Housing Price Dataset

Target Variable:

* Price

Selected Features:

* Rooms
* Distance
* Bathroom
* Car
* Longtitude
* Lattitude
* YearBuilt

Rows containing missing values were removed during preprocessing.

---

## Project Structure

```text
melbourne-linear-regression/
│
├── melb_data.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_test_split.py
│   ├── linear_regression.py
│   └── evaluation.py
│
├── images/
│   └── convergence_plot.png
│
├── report/
│   └── report.pdf
│
├── main.py
├── README.md
└── .gitignore
```

---

## Methodology

### 1. Data Loading

The dataset is loaded using Python's built-in CSV module.

### 2. Data Cleaning

Rows containing missing values are removed.

### 3. Feature Scaling

Features are normalized to improve convergence speed during Gradient Descent.

### 4. Train-Test Split

The dataset is split into:

* 80% Training Data
* 20% Testing Data

### 5. Linear Regression

The prediction model is:

ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

### 6. Cost Function

Mean Squared Error (MSE) is used as the loss function.

### 7. Optimization

Gradient Descent is used to learn model parameters.

---

## Evaluation Metrics

The model is evaluated using:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Results

| Metric        | Value      |
| ------------- | ---------- |
| Learning Rate |0.1         |
| Epochs        |1001        |
| Final MAE     |298011.53896|
| RMSE          |414303.73288|
| R² Score      |0.544343    |
| Training Time |14.30373    |

---

## Cost Function Convergence

The following graph illustrates the decrease in cost over training epochs.

![alt text](<Cost vs Time for pure python model for lineat regression-1.png>)

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/kushagratripathi488-stack/melbourne-housing-linear-regression-with-pure-python/commits/main/
```

Navigate to the project directory:

```bash
cd melbourne-linear-regression
```

Run the project:

```bash
python main.py
```

---

## Key Learnings

* Implementation of Multivariable Linear Regression from scratch
* Understanding Gradient Descent optimization
* Data preprocessing techniques
* Feature scaling and normalization
* Model evaluation using regression metrics
* Git and GitHub workflow for machine learning projects

---

## Future Improvements

* Mini-batch Gradient Descent
* Stochastic Gradient Descent
* L2 Regularization
* Feature Engineering
* Hyperparameter Tuning
* Comparison with NumPy and Scikit-Learn implementations

---

## Author

Kushagra Tripathi

GitHub:@kushagra3148
