
# coding: utf-8

# # Assignment 1
# This jupyter notebook is meant to be used in conjunction with the full questions in the assignment pdf.
# 
# ## Instructions
# - Write your code and analyses in the indicated cells.
# - Ensure that this notebook runs without errors when the cells are run in sequence.
# - Do not attempt to change the contents of the other cells.
# 
# ## Submission
# - Ensure that this notebook runs without errors when the cells are run in sequence.
# - Rename the notebook to `<roll_number>.ipynb` and submit ONLY the notebook file on moodle.

# ### Environment setup
# 
# The following code reads the train and test data (provided along with this template) and outputs the data and labels as numpy arrays. Use these variables in your code.
# 
# ---
# #### Note on conventions
# In mathematical notation, the convention is tha data matrices are column-indexed, which means that a input data $x$ has shape $[d, n]$, where $d$ is the number of dimensions and $n$ is the number of data points, respectively.
# 
# Programming languages have a slightly different convention. Data matrices are of shape $[n, d]$. This has the benefit of being able to access the ith data point as a simple `data[i]`.
# 
# What this means is that you need to be careful about your handling of matrix dimensions. For example, while the covariance matrix (of shape $[d,d]$) for input data $x$ is calculated as $(x-u)(x-u)^T$, while programming you would do $(x-u)^T(x-u)$ to get the correct output shapes.

# In[410]:


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)


# # Questions
# ---
# ## 1.3.1 Representation
# The next code cells, when run, should plot the eigen value spectrum of the covariance matrices corresponding to the mentioned samples. Normalize the eigen value spectrum and only show the first 100 values.

# In[411]:


# Samples corresponding to the last digit of your roll number (plot a)
import random
import matplotlib.pyplot as plt

random3k=  np.random.choice(6000,(3000,1))
random3k = random3k.reshape(3000)
random3k.shape

def minmaxnorm(x):
    return (x-x.min())/(x.max()-x.min())

class getparameter:
    
    def preparedataset(self,data,labels,digit):
        if digit==100:
            data_labels=data
        elif digit == 50:
            data_labels=data[random3k]
        else:
            index = np.where(labels==digit)
            data_labels = data[index]
        data_labels = data_labels.T    
        return data_labels
    
    def calculatemeancoveigen(self,dataset):
        mean1= np.mean(dataset,axis=1)
        std1  = np.std(dataset,axis=1)
        cov1 = np.cov(dataset)
        egv1 = np.real(np.linalg.eigvals(cov1))
        return mean1,std1,cov1,egv1
    
myparameter = getparameter()

mean0,std0,cov0,eigenval0 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,0))
mean1,std1,cov1,eigenval1 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,1))
mean2,std2,cov2,eigenval2 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,2))
mean3,std3,cov3,eigenval3 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,3))
mean4,std4,cov4,eigenval4 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,4))
mean5,std5,cov5,eigenval5 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,5))
mean6,std6,cov6,eigenval6 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,6))
mean7,std7,cov7,eigenval7 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,7))
mean8,std8,cov8,eigenval8 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,8))
mean9,std9,cov9,eigenval9 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,9))

eigenval_norm8 = minmaxnorm(eigenval8)
plt.figure(figsize = (5,5))
plt.plot(eigenval_norm8[:100],label = "Class8")
plt.title("Eigen Value Plot")
plt.legend()


# In[412]:


# Samples corresponding to the last digit of your roll number (plot a)

#mean9,std9,cov9,eigenval9 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,9))
eigenval_norm9 = minmaxnorm(eigenval9)
plt.figure(figsize = (5,5));
plt.plot(eigenval_norm9[:100],label = "Class9")
plt.title("Eigen Value Plot")
plt.legend()


# In[413]:


# All training data (plot c)
mean100,std100,cov100,eigenval100 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,100))
eigenval_norm100 = minmaxnorm(eigenval100)

plt.plot(eigenval_norm100[:100],label = "Full data")
plt.title("Eigen Value Plot")
plt.legend()


# In[414]:


# Randomly selected 50% of the training data (plot d)
mean50,std50,cov50,eigenval50 = myparameter.calculatemeancoveigen(myparameter.preparedataset(train_data,train_labels,50))
eigenval_norm50 = minmaxnorm(eigenval50)

plt.plot(eigenval_norm50[:100],label = "50% data")
plt.title("Eigen Value Plot")
plt.legend()


# ### 1.3.1 Question 1
# - Are plots a and b different? Why?
# - Are plots b and c different? Why?
# - What are the approximate ranks of each plot?

# ---
# Your answers here (double click to edit)
# Plots a and b are almost similar, because their individual convariance matrix (for their own class) is almost same.
# 
# Plots b and c are bit different if we watch closely their eigenvalues' convergance plot. plot c has full data, which consists of all classes' data.So, covariance matrix of c will be slightly different for convariance matrix of a particular class 9.
# 
# Approximate rank of each plot looks like arond 100, as if we see the plot, y dimension touches to 0 after 100 eigenvalues' plot. So, no of eigenvalues are almost 100. Hence, rank of each plot is approximate 100.
# 
# ---

# ### 1.3.1 Question 2
# - How many possible images could there be?
# - What percentage is accessible to us as MNIST data?
# - If we had acces to all the data, how would the eigen value spectrum of the covariance matrix look?

# ---
# Your answers here (double click to edit)
# 1. total number of possible image will be 2^784
# 2. Percentage data available to us as MNIST data = (7000/2^784)*100 = 6.8*10^(-231) percent
# 3. If we had access to all available data, and we plot eigen specturm of the co-varaiance matrix, eigen spectrum will be a line, because variance will be same in all directions. 
# ---

# ## 1.3.2 Linear Transformation
# ---
# ### 1.3.2 Question 1
# How does the eigen spectrum change if the original data was multiplied by an orthonormal matrix? Answer analytically and then also validate experimentally.

# ---
# Analytical answer here (double click to edit)
# 
# If we multiply original data by an orthonormal matrix, eigen spectrum will remain same. When we transform any matrix by orthonormal matrix, original matrix either ramains same (if orthonormal matrix is Identity matrix) or their axis is rotated by some angle , but in both the cases number of independent columns/rows of original matrix remains same and so is there eigen values. Hence, eigen value spectrum will not be affected.
# 
# Below plot shows original and transformed matrix have same eigenvalue spectrum.
# 
# ---

# In[415]:


# Experimental validation here.
# Multiply your data (train_data) with an orthonormal matrix and plot the
# eigen value specturm of the new covariance matrix.

import numpy as np    

def rvs(dim):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H
ortho784= rvs(784)
#from scipy.stats import ortho_group

train_data.shape
data_transform = np.dot(train_data,ortho784)
#calculate and plot eigenvalue spectrum for full transformed data
meanT100,stdT100,covT100,eigenvalT100 = myparameter.calculatemeancoveigen(myparameter.preparedataset(data_transform,train_labels,100))
eigenval_normT100 = minmaxnorm(eigenvalT100)

plt.plot(eigenval_norm100[:100],label = "Full data")
plt.plot(eigenval_normT100[:100],label = "Full transformed data")

plt.title("Eigen Value Plot")
plt.legend()


# ### 1.3.2 Question 2
# If  samples  were  multiplied  by  784 × 784  matrix  of rank 1 or 2, (rank deficient matrices), how will the eigen spectrum look like?

# ---
# Your answer here (double click to edit)
# In that case, transformed matrix of 6000 samples will have only 2 dimensions left, rest all dimensions will be dropped by this rank deficient matrix. Eigen value spectrum will have only 2 eigen values plotted.
# ---

# ### 1.3.2 Question 3
# Project the original data into the first and second eigenvectors and plot in 2D

# In[397]:


# Plotting code here

_,eigvec100  = np.linalg.eig(cov100)
eigvec100 = np.real(eigvec100[:,0:2])
eigproject = np.dot( eigvec100.T,train_data.T)

plt.scatter(eigproject[0,:],eigproject[1,:],label = "Full data")
plt.title("Eigen Value Plot in 2D for first 2 eigen values")
plt.legend()


# ## 1.3.3 Probabilistic View
# ---
# In this section you will classify the test set by fitting multivariate gaussians on the train set, with different choices for decision boundaries. On running, your code should print the accuracy on your test set.

# In[441]:


# Print accuracy on the test set using MLE
import pandas as pd
from scipy.stats import norm

def accuracyscore(actual,pred):
    pred = pred.astype('int16')
    actual = actual.astype('int16')
    count = 0

    for i in range(actual.shape[0]):
        if pred[i] == actual[i]:
            count = count + 1
    return  count/actual.shape[0]
def numpymode(a):
    (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    return a[index]

mle_data = pd.DataFrame(data = test_labels,columns = ["actual"])
list1 = [mean0,mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8,mean9]
list2 = [std0,std1,std2,std3,std4,std5,std6,std7,std8,std9]
mle_data["predclass"]=0


for i in range(test_data.shape[0]):
    prob=[]
    for j in range(10):
        feature_prob = norm.logpdf(test_data[i].reshape(1,784),list1[j].reshape(1,784),list2[j].reshape(1,784))
        prob.append(np.nansum(feature_prob))
    mle_data.loc[i,"predclass"] = [p for p in range(len(prob)) if prob[p]== np.max(prob)][0]

mle_data["actual"].astype(int)
print("Accuracy Score of MLE is ",accuracyscore(mle_data["actual"], mle_data["predclass"]))


# In[442]:


# Print accuracy on the test set using MAP
# (assume a reasonable prior and mention it in the comments)
# Prior probablity is assumed as 600/6000= 0.1 for each class in train data, with Same prior , MAP and MLE are same
map_data = pd.DataFrame(data = test_labels,columns = ["actual"])
list1 = [mean0,mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8,mean9]
list2 = [std0,std1,std2,std3,std4,std5,std6,std7,std8,std9]
map_data["predclass"]=0
prior_prob=0.1

for i in range(test_data.shape[0]):
    prob=[]
    for j in range(10):
        feature_prob = norm.logpdf(test_data[i].reshape(1,784),list1[j].reshape(1,784),list2[j].reshape(1,784)) 
        prob.append(np.nansum(feature_prob)+np.log(0.1))
    map_data.loc[i,"predclass"] = [p for p in range(len(prob)) if prob[p]== np.max(prob)][0]

map_data["actual"].astype(int)
print("Accuracy Score of MAP is ",accuracyscore(map_data["actual"], map_data["predclass"]))


# In[443]:


# Print accuracy using Bayesian pairwise majority voting method
bay_data = pd.DataFrame(data = test_labels,columns = ["actual"])
list1 = [mean0,mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8,mean9]
list2 = [cov0,cov1,cov2,cov3,cov4,cov5,cov6,cov7,cov8,cov9]
num=0
for j in range(10):
    for k in range(10):
        if k>j:
            bay_data["pair{0}".format(num)]=0
            num=num+1
            if num==45:
                bay_data["predclass"] = 0
            inv_covmean = np.linalg.pinv((list2[j] + list2[k])/2)
            m1 = list1[j]
            m2 = list1[k]
            w = np.dot(2*inv_covmean,(m1-m2))
            c = np.dot(np.dot(m2.T,inv_covmean),m2) - np.dot(np.dot(m1.T,inv_covmean),m1)
            for i in range(1000):
                if np.dot(test_data[i].reshape(784,1).T,w.reshape(784,1))[0][0]+ c > 0:
                    bay_data.iloc[i,num] = j
                else:
                    bay_data.iloc[i,num] = k
                if num==45:
                    bay_data.loc[i,"predclass"]= numpymode(bay_data.iloc[i,1:46].astype("int16"))
print("Accuracy of Bayesian pairwise majority voting method is : ",accuracyscore(bay_data["actual"], bay_data["predclass"]))


# In[444]:


# Print accuracy using Simple Perpendicular Bisector majority voting method
pb_data = pd.DataFrame(data = test_labels,columns = ["actual"])
list1 = [mean0,mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8,mean9]

for i in range(test_data.shape[0]):
    class1=[]
    for j in range(10):
        for k in range(10):
            if k>j:
                vec1 = (list1[j]-list1[k]).reshape(1,784)
                vec2 = (test_data[i]- (list1[j]+list1[k])/2).reshape(784,1)
                WTX = np.dot(vec1,vec2)
                if WTX > 0:
                    class1.append(j)
                else:
                    class1.append(k)
    pb_data.loc[i,"predclass"] = numpymode(class1)
print("Accuracy of  perpendicular bisector is : ",accuracyscore(pb_data["actual"], pb_data["predclass"]))
    


# ### 1.3.3 Question 4
# Compare performances and salient observations

# ---
# Your analysis here (double click to edit)
# Performance/Accuracy of Bayesian pairwise majority voting is best among above 4 linear classifier as covariance matrix are same in this case. 
# MLE and MAP are same (since prior probablity is same for all 10 classes) and are worst in accuracy among 4 classifiers.
# 
# ---

# ## 1.3.4 Nearest Neighbour based Tasks and Design
# ---
# ### 1.3.4 Question 1 : NN Classification with various K
# Implement a KNN classifier and print accuracies on the test set with K=1,3,7

# In[440]:


# Your code here
# Print accuracies with K = 1, 3, 7
knn=[1,3,7]
for k in knn:
    knn_data = pd.DataFrame(data = test_labels,columns = ["actual"])
    for i in range(1000):
        distance=[]
        diff = (train_data - test_data[i].reshape(1,784))
        distance = np.sum(abs(diff),axis=1)
        sorted_index = np.argsort(distance)[0:k]
        predclass = train_labels[sorted_index]
        knn_data.loc[i,"predclass"] = numpymode(predclass.astype("int16"))
    print("Accuracy of KNN for k={} is : ".format(k),accuracyscore(knn_data["actual"], knn_data["predclass"]))


# ### 1.3.4 Question 1 continued
# - Why / why not are the accuracies the same?
# - How do we identify the best K? Suggest a computational procedure with a logical explanation.

# ### 1.3.4 Question 2 :  Reverse NN based outlier detection
# A sample can be thought of as an outlier is it is NOT in the nearest neighbour set of anybody else. Expand this idea into an algorithm.

# In[246]:


# This cell reads mixed data containing both MNIST digits and English characters.
# The labels for this mixed data are random and are hence ignored.
mixed_data, _ = read_data("outliers.csv")
print(mixed_data.shape)


# ---
# Your analysis here (double click to edit)
# 
# Accuracy is different for different value of k. Since, we are looking for k number of nearest neighbours. So, for example if k=1 and our classifier will look for nearest neighbour and if its wrong, it will impact model's accuracy. 
# A small value of k means that noise will have a higher influence on the result and a large value make it computationally expensive.
# So, better to check for some value of k like 3 or 5  and predicts mode of all predicted values. In this sample, k=3 gives us the best result. We can plot a graph 'k' vs 'accuracy'. Take some k value after which if you see that increasing k value does not give better accuracy. It will be an elbow curve. Take elbow point as best k for the classifier.
# 
# ---

# In[445]:


list1 = [mean0,mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8,mean9]
for i in range(20):
    mindlist = []
    for j in range(10):
        diff1 = (list1[j] - mixed_data[i]).reshape(1,784)
        distance1 = np.sum(abs(diff1),axis=1)
        mindlist.append(distance1[0])
    sorted_index = np.argsort(mindlist)[0:5]
    mindistelement = list1[sorted_index[0]]
    mindistance = mindlist[sorted_index[0]]
    maxdlist = []
    for k in range(10):
        diff = (list1[k] - mindistelement).reshape(1,784)
        distance = np.sum(abs(diff),axis=1)
        maxdlist.append(distance[0])
    sorted_index1 = np.argsort(maxdlist)[0:10]
    maxdistance = maxdlist[sorted_index1[-1]]
    
    if mindistance > maxdistance:
        print("Row {} of mixed dataset is an outlier".format(i))
    
    
        


# ### 1.3.4 Question 3 : NN for regression
# Assume that each classID in the train set corresponds to a neatness score as:
# $$ neatness = \frac{classID}{10} $$
# 
# ---
# Assume we had to predict the neatness score for each test sample using NN based techiniques on the train set. Describe the algorithm.

# ---
# Your algorithm here (double click to edit)
# 
# ---

# ### 1.3.4 Question 3 continued
# Validate your algorithm on the test set. This code should print mean absolute error on the test set, using the train set for NN based regression.

# In[446]:


# Your code here
neatness_data = pd.DataFrame(data = test_labels*0.1,columns = ["neatnessactual"])
    
k= 3
for i in range(test_data.shape[0]):
    distance=[]
    diff = (train_data - test_data[i].reshape(1,784))
    distance = np.sum(abs(diff),axis=1)
    sorted_index = np.argsort(distance)[0:k]
    neatness = (train_labels[sorted_index].sum())*0.1/k
    neatness_data.loc[i,"neatnesspred"] = neatness
    
neatness_mae = np.sum(np.power((neatness_data["neatnesspred"]-neatness_data["neatnessactual"]),2))
print("Neatness mean absolute square error for k = {0} is : ".format(k), neatness_mae)
    


# ---
# # FOLLOW THE SUBMISSION INSTRUCTIONS
# ---
