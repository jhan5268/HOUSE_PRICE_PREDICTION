# House Price Prediction Model 🏠💰

An end-to-end Machine Learning regression project designed to predict residential real estate prices based on structural, environmental, and regional attributes. 

## 🚀 Key Features
* **Data Cleansing**: Handled missing values, treated extreme outliers, and corrected skewed data distributions.
* **Feature Engineering**: Encoded categorical location data, scaled numerical inputs, and analyzed feature correlations.
* **Model Evaluation**: Implemented and benchmarked multiple regression algorithms to maximize predictive accuracy.
* **Feature Importance**: Extracted key market drivers to discover which property elements impact value the most.

## 🛠️ Tech Stack
* **Language**: Python
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn
* **Visualization**: Matplotlib, Seaborn
* **Environment**: Jupyter Notebook / VS Code

## 📂 Project Structure
```text
├── data/                  # Raw Kaggle CSV files (train/test) and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and Model Training
├── src/                   # Python scripts for data pipeline and modeling
├── .gitignore             # Python ignore file
├── README.md              # Project documentation
└── requirements.txt       # Dependencies list
```

## 📈 Performance Summary
The final tuned regression model achieved a strong $R^2$ score, minimizing Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to deliver reliable, data-driven pricing estimates.

## 🧑‍💻 How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com
   cd HOUSE_PRICE_PREDICTION
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/house_price_prediction.ipynb
   ```
