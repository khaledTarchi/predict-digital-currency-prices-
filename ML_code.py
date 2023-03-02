import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score



# Read CSV file
data = pd.read_csv('C:/Users/Tech/Documents/My project-python/code/algorand.csv')

data=data.drop(["coin_name"],axis=1)







# convert date to numirical values(int)

data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].astype(np.int64) // 10 ** 9



# visualization
Relation=data.corr()
sns.heatmap(Relation)
sns.pairplot(data)





# variable,target
X = data[['date', 'total_volume','market_cap']].values
Y = data['price'].values





# Define training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)






poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, Y_train)


# predict 
Y_pred = poly_reg.predict(X_poly_test)



# Score calculate
score = r2_score(Y_test, Y_pred)
print("Score: ",score)



# inputs

date = input("Enter a date in yyyy-mm-dd format: ")
total_volume = input("Enter a total volume value: ")
market_cap=input("Enter a market cap:")

data={"date":[date],"total_volume":total_volume,'market_cap':market_cap}

data=pd.DataFrame(data)

data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].astype(np.int64) // 10 ** 9


user_input = poly.transform(np.array(data))
predicted_price = poly_reg.predict(user_input)
print("\n\n\n    Predicted Price:", predicted_price[0])




























