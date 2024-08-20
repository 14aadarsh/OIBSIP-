#Sales pridiction with machine learning using python

#Importing Required Libraries
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#load data
data = pd.read_csv('OSISTASK5.csv')

#Visualizing
sns.pairplot(data)
plt.show()

#Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True,
cmap='coolwarm')
plt.show()
  
#Features
X = data[['TV', 'Radio', 'Newspaper']]

#Target
y = data['Sales'] 

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = LinearRegression()
model.fit(X_train, y_train)

#Model Prediction
y_pred = model.predict(X_test)

#Model Evaluation
print(f"R² score: {r2_score(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")

#Visualization
plt.figure(figsize = (10,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales - Linear Regression")
plt.show()
