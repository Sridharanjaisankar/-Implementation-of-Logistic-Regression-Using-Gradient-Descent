# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.
## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sridharan J
RegisterNumber: 212222040158  
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter=',')
X = data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
X_plot= np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta, X, y):
  h=sigmoid(np.dot(X, theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J, grad
  
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
  
def gradient(theta, X,y):
  h = sigmoid(np.dot(X, theta))
  grad = np.dot(X.T, h-y) / X.shape[0]
  return grad
  
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res=optimize.minimize(fun=cost, x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output:
## 1.Array Value of x:
![image](https://github.com/SOMEASVAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93434149/55cb2d2f-1dac-4094-af0b-62e2ba7df4a0)

## 2.Array Value of y:
![image](https://github.com/SOMEASVAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93434149/75fdb449-4986-4b20-b02b-05f53e6bcbf4)

## 3.Exam 1 - score graph:
![image](https://github.com/SOMEASVAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93434149/b3cbe8c4-a5cc-400c-bd5c-3dfd37b68361)

## 4.Sigmoid function graph:
![image](https://github.com/SOMEASVAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93434149/6ccae71f-c3ff-4ce6-a1ee-9c3c3f3ba141)

## 5.X_train_grad value:
![image](https://github.com/SOMEASVAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93434149/5b19fe6c-c89f-49ba-afc0-0e663dae91db)

## 6.Y_train_grad value:
![image](https://github.com/SOMEASVAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93434149/6bda78a3-857f-4275-9129-a1333547076d)

## 7.Print res.x:
![image](https://github.com/SOMEASVAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93434149/90b13b32-f78d-4937-9193-55470a284305)

## 8.Decision boundary - graph for exam score:
![image](https://github.com/SOMEASVAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93434149/df5585b8-11c2-4899-a351-0c0d1ac80f4e)

## 9.Proability value:
![image](https://github.com/SOMEASVAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93434149/cae47c95-e330-464d-94d7-ea4b2b0c55f8)

## 10.Prediction value of mean:
![image](https://github.com/SOMEASVAR/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93434149/1cc26a76-71f5-4713-a728-bf5861b69ee5)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

