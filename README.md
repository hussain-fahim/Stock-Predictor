# 📈 Stock Price Prediction using LSTM

This project builds and trains a **Long Short-Term Memory (LSTM)** neural network to predict stock prices using historical closing data fetched directly from **Yahoo Finance**.  
It demonstrates how deep learning can be applied to **time series forecasting** for financial data.

---

## 🧠 Overview

The model:
- Downloads historical stock data for any given ticker symbol (e.g., **AAPL**, **TSLA**, **GOOGL**, **MSFT**) using the `yfinance` API  
- Prepares **windowed datasets** for sequence learning (3-day lookback by default)  
- Trains a **Sequential LSTM model** with multiple dense layers  
- Splits the data into **training**, **validation**, and **testing** sets  
- Visualizes actual vs. predicted closing prices  
- Performs **recursive multi-step future predictions**

---

## ⚙️ Technologies Used

| Library | Purpose |
|----------|----------|
| `yfinance` | Fetching historical stock data |
| `pandas` | Data manipulation and time indexing |
| `numpy` | Numerical array operations |
| `matplotlib` | Plotting and visualization |
| `tensorflow.keras` | Building and training the LSTM model |
| `datetime` | Handling date/time operations |
| `copy` | Deep copying arrays for recursive prediction |

---

## 📂 Project Structure

```
stock-predictor-lstm/
│
├── stock_predictor.py         # Main LSTM training and prediction script
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
```

---

## 🧩 Data Preparation

1. **Download historical stock data:**
   ```python
   import yfinance as yf
   stock = yf.Ticker("AAPL")  # Replace with any ticker symbol
   df = stock.history(period="max")
   ```

2. **Use only the closing price:**
   ```python
   df = df[['Close']]
   ```

3. **Convert the date column to datetime objects** for time-based indexing.

4. **Create windowed dataframes** with a rolling lookback of 3 days (e.g., `[t-3, t-2, t-1] → [t]`):  
   ```python
   df_to_windowed_df(df, first_date_str, last_date_str, n=3)
   ```

---

## 🧱 Model Architecture

| Layer | Type | Details |
|--------|------|----------|
| Input | — | Sequence of 3 timesteps × 1 feature |
| LSTM | 64 units | Learns temporal dependencies |
| Dense | 32 units, ReLU | Hidden fully connected layer |
| Dense | 32 units, ReLU | Hidden fully connected layer |
| Dense | 1 unit | Predicts next closing price |

**Loss Function:** Mean Squared Error (MSE)  
**Optimizer:** Adam (learning rate = 0.001)  
**Metric:** Mean Absolute Error (MAE)

---

## 🏋️‍♂️ Training

The dataset is split as:
- **80% → Training**
- **10% → Validation**
- **10% → Testing**

Train the model with:
```python
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100
)
```

---

## 📊 Evaluation and Visualization

After training, the script:
- Predicts stock prices on **training**, **validation**, and **testing** datasets  
- Plots actual vs. predicted closing prices  
- Generates a **recursive forecast** extending beyond known data  

Example visualization (schematic):

```
Training Predictions — Blue
Validation Predictions — Orange
Testing Predictions — Green
Recursive Predictions — Red
```

To display the plots, uncomment any of the `plt.show()` lines in your code.

---

## 🔁 Recursive Prediction Logic

The code includes a section that performs **recursive, multi-step forecasting**:

```python
recursive_predictions = []
recursive_dates = np.concatenate([dates_val, dates_test])

for target_date in recursive_dates:
    last_window = deepcopy(X_train[-1])
    next_prediction = model.predict(np.array([last_window])).flatten()
    recursive_predictions.append(next_prediction)
    last_window[-1] = next_prediction
```

This simulates predicting multiple future days by feeding each new prediction back into the input window.

---

## 🧩 Example Output

| Dataset | MSE | MAE |
|:---------|:----:|:----:|
| Training | ~0.0003 | ~0.016 |
| Validation | ~0.0005 | ~0.021 |
| Testing | ~0.0006 | ~0.023 |

*(Metrics are illustrative — actual values depend on the training run and chosen ticker.)*

---

## 📈 Example Plot

![Stock Prediction Graph](assets/example_plot.png)

*(The plot compares real vs. predicted stock prices over time.)*

---

## 📦 Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/stock-predictor-lstm.git
cd stock-predictor-lstm
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the predictor
```bash
python stock_predictor.py
```

### 4️⃣ Customize the ticker
Inside your Python script, change:
```python
stock = yf.Ticker("TSLA")  # Example: Tesla
```
to any other valid ticker symbol (e.g., "AAPL", "GOOGL", "AMZN", "NVDA", etc.).

---

## 🧠 Requirements

Add this to your `requirements.txt`:
```
yfinance
pandas
numpy
matplotlib
tensorflow
```

---

## 🚀 Future Improvements

- Normalize input data with `MinMaxScaler` for smoother convergence  
- Extend lookback window dynamically  
- Add **dropout layers** for regularization  
- Tune hyperparameters (batch size, epochs, units)  
- Implement early stopping and checkpointing  
- Deploy via a **Flask API** or **Streamlit dashboard**  
- Expand to predict multiple tickers simultaneously  

---
> *“The goal is not to predict the future perfectly — but to be less surprised when it arrives.”*
