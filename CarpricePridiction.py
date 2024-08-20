#Car Price Pridiction Using Python 

#Importing Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load the Data
data = pd.read_csv('OSISTASK3.csv')

#print first few rows to understand data
print(data.head())

#check for missing values
print(data.isnull().sum())

#i'm assuming we keep all numerical columns
numerical_data = data.select_dtypes(include=['float64','int64'])

#visualizing 
sns.pairplot(numerical_data)
plt.show()

#Heatmap
sns.heatmap(numerical_data.corr(),annot=True,
cmap='coolwarm')
plt.show()

#Features 
X = data[['Year', 'Selling_Price', 'Present_Price','Driven_kms',
          'Owner']]
#Target
y = data['Selling_Price']

#Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Training 
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
plt.xlabel("Actual Selling  Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Price - Linear Regression")
plt.show()