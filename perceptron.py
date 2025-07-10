from sklearn import datasets                          # import iris dataset using datasets from sklearn
from sklearn.model_selection import train_test_split  # splitting data into training set and testing set
from sklearn.linear_model import Perceptron           # importing the Perceptron Algorithm from sklearn.linear_model 
from sklearn.metrics import accuracy_score            # evaluating model using accuracy metric

# Link to dataset: https://archive.ics.uci.edu/ml/datasets/iris


iris_dataset = datasets.load_iris()                   # load the iris dataset
X = iris_dataset.data[:, [2,3]]                       # load the data in indices 2 and 3 (petal length and petal width) into the variable X
y = iris_dataset.target                               # loads the target data into variable y



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)   # splitting the data into training data and testing data


perceptron_model = Perceptron(max_iter=3, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=True)



perceptron_model.fit(X_train, y_train)            # train the model on training data



y_predict = perceptron_model.predict(X_test)      # test the model on test data



accuracy = accuracy_score(y_test, y_predict)  # determine accuracy
accuracy = accuracy * 100                     # convert to percentage

print("\nAccuracy of model: " + str(round(accuracy,2)) + "%\n") # print accuracy