#data loading and inspection
#-----------------------------
import pandas as pd
import numpy as np

# Load your Kaggle files
train_df = pd.read_csv('../data/train.csv')
print(train_df.head())
print(train_df.info())

#exploratory data analysis
#------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Plot target variable distribution (Checking for skewness)
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Distribution of House Prices')
plt.show()

# Correlation matrix heatmap to find strongest price drivers
plt.figure(figsize=(12, 8))
sns.heatmap(train_df.corr(numeric_only=True), cmap='coolwarm')
plt.show()

#data cleaning and feature engineering
#---------------------------------------
from sklearn.preprocessing import StandardScaler

# Fill missing numerical data with the median value
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(train_df['LotFrontage'].median())

# Separate features (X) and target variable (y)
X = train_df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageCars']] # Example features
y = train_df['SalePrice']

# Scale features for balanced model interpretation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#model training and evalation\
#---------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Linear Regression baseline model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values and calculate performance metrics
predictions = model.predict(X_val)
print(f"R2 Score: {r2_score(y_val, predictions):.4f}")
print(f"Mean Absolute Error (MAE): ${mean_absolute_error(y_val, predictions):.2f}")

#feature importance extraction
#---------------------------------
# Check which features impact the pricing engine the most
importance = model.coef_
for i, v in enumerate(importance):
    print(f'Feature: {X.columns[i]}, Score: {v:.2f}')
