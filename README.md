# 📈 Revenue Forecasting Using an Adaptive Hybrid Time Series Framework (30-Day Horizon)

## 📌 Overview

Accurate revenue forecasting is essential for business planning, budgeting, inventory management, and strategic decision-making. This project implements an **end-to-end time series forecasting system** to predict **daily revenue for the next 30 days** using historical data.

Instead of relying on a single forecasting technique, we propose an **Adaptive Hybrid Forecasting Framework** that combines:
- A **statistical time series model (ARIMA)** to capture trend and seasonality,
- A **machine learning regression model** trained on engineered time-series features to learn nonlinear patterns,
- A **dynamic fusion strategy** that weights each model based on recent performance.

This hybrid approach improves robustness and stability, especially when revenue trends change over time.

---

## 🎯 Objectives

- Forecast daily revenue for the next **30 days**.
- Build and compare multiple models:
  - Baseline forecasting model (Naive / Moving Average)
  - ARIMA model
  - Machine Learning regression model
  - Proposed Adaptive Hybrid model
- Evaluate all models using **RMSE** and **MAE**.
- Visualize:
  - Historical revenue trends
  - Actual vs predicted values on test data
  - 30-day future revenue forecast
- Provide an interactive visualization interface using **Streamlit**.

---

## 🗂️ Dataset Description

The dataset contains historical daily revenue records with the following columns:
- `customer ID` : Customer of that day
- `date`: Date of observation
- `revenue`: Revenue value for that day

### Data Preprocessing Steps

1. Convert the `date` column into datetime format.
2. Sort the data chronologically to maintain time order.
3. Check for missing dates and missing values.
4. Resample the data to daily frequency if required.
5. Fill missing values using forward fill or interpolation methods.
6. Perform basic exploratory data analysis to identify:
   - Overall trend
   - Seasonality patterns
   - Outliers or abnormal spikes

This preprocessing ensures the data is suitable for both statistical modeling and machine learning.

---

## 🧠 Methodology

### 1. Baseline Forecasting Model

A simple baseline model is implemented to serve as a reference point:
- **Naive Forecast**: Assumes the next value is equal to the last observed value.
- **Moving Average Forecast**: Predicts future values as the average of the last *k* days.

This helps measure the improvement achieved by more advanced models.

---

### 2. Statistical Time Series Model: ARIMA

ARIMA (AutoRegressive Integrated Moving Average) is used to model linear time series patterns such as trend and seasonality.

**Steps:**
1. Check stationarity of the series and apply differencing if required.
2. Select ARIMA parameters (p, d, q) using standard techniques or automated search.
3. Train the ARIMA model on the training portion of the dataset.
4. Generate predictions for:
   - The test period (last 30 days),
   - The future 30-day horizon.
5. Store predictions for evaluation and hybrid fusion.

ARIMA is effective in modeling structured temporal patterns present in revenue data.

---

### 3. Machine Learning Regression Model

To capture nonlinear relationships, the time series forecasting problem is transformed into a supervised learning problem using engineered features.

#### Feature Engineering

The following features are created from historical revenue data:

- **Lag Features:**
  - `lag_1`: Revenue at time t-1 (previous day)
  - `lag_7`: Revenue at time t-7 (one week ago)
  - `lag_14`: Revenue at time t-14 (two weeks ago)

- **Rolling Statistics:**
  - Rolling mean over 7 and 14 days (captures local trend level)
  - Rolling standard deviation over 7 and 14 days (captures volatility)

- **Trend Feature:**
  - Rolling mean slope to capture the direction and strength of the trend

- **Optional Calendar Features:**
  - Day of week
  - Weekend indicator

#### Model Training

1. Perform a time-based train-test split.
2. Train a regression model (e.g., Linear Regression or Random Forest) using the engineered features.
3. Generate predictions for the test period and future horizon.
4. Store predictions for evaluation and hybrid fusion.

---

### 4. Trend Regime Detection

Revenue behavior often changes over time (growth, stability, decline). To capture this:

1. Compute a rolling mean of revenue.
2. Compute the slope of the rolling mean.
3. Classify each period into:
   - **Uptrend** (positive slope),
   - **Stable** (near-zero slope),
   - **Downtrend** (negative slope).

This analysis helps understand changing revenue patterns and supports adaptive forecasting.

---

### 5. Adaptive Hybrid Forecasting (Proposed Method)

Instead of selecting a single model, predictions from ARIMA and the ML model are combined using a **dynamic weighting strategy**:

1. Evaluate both models on a recent validation window.
2. Compute error metrics (RMSE or MAE) for each model.
3. Assign weights inversely proportional to their errors:

----

## 📊 Visualization and User Interface

An interactive dashboard is built using **Streamlit** to:
- Display historical revenue trends
- Compare actual vs predicted values on the test period
- Show the 30-day future revenue forecast
- Display evaluation metrics for all models

This makes the results easy to interpret for both technical and non-technical users.

----

## 🏗️ Project Structure

.
├── data/
│ └── revenue.csv
├── model.py # Data processing, feature engineering, training, forecasting
├── app.py # Streamlit UI and visualization
├── requirements.txt # Python dependencies
├── README.md


---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Statsmodels, Scikit-learn, Matplotlib  
- **User Interface:** Streamlit  
- **Version Control:** GitHub  


