import pandas as pd                                             # import for dataframes
import numpy as np                                              # import for arrays
import matplotlib.pyplot as plt                                 # import for plotting
from sklearn import linear_model                                # importing the Linear Regression Algorithm from sklearn
from sklearn.model_selection import train_test_split            # splitting data into training set and testing set  


# Link to dataset: https://archive.ics.uci.edu/ml/datasets/student+performance


data = pd.read_csv("student-mat.csv", sep=";")                      # Read in the data from the csv file
print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] # extract features to be used for this model
print(data.head())

target = "G3"
X = np.array(data.drop([target], 1))
Y = np.array(data[target])



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)



linear_regression_model = linear_model.LinearRegression()



linear_regression_model.fit(x_train, y_train)



y_predict = linear_regression_model.predict(x_test)



accuracy = linear_regression_model.score(x_test, y_test)
accuracy = accuracy * 100
accuracy = accuracy.round(2)

print(f"\nAccuracy of the model is: {accuracy}%\n")


for i in range(len(y_predict)):
  print(y_predict[i].round(2), x_test[i], y_test[i])



x = "G2"
y = "G3"
plt.scatter(data[x], data[y])
plt.xlabel(x)
plt.ylabel("Final Grade")
plt.show()