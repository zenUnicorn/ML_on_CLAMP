# Machine Learning on CLAMP Dataset
Supervised and Unsupervised Machine Learning on CLAMP Dataset


## Introduction
Malicious programs or malware is an intentionally written program to indulge in various malicious activities, ranging from stealing user’s information to 
cyber-espionage. The behavioral dynamism exposed by the malware is dependent on various factors such as nature of the attack, sophisticated technology and 
the rapid increase in exploitable vulnerabilities. Malware attacks also increased along with the rapid growth in the use of digital devices and internet. 
The exponential increase in the creation of new malware in the last five years has made malware detection a challenging research issue.


```python
# Programming language


Built using python
```
----
## PACKAGE VERSIONS

- pandas==1.3.5
- numpy==1.21.6
- scikit-learn==1.0.2

```python
!pip install pandas==1.3.5 numpy==1.21.6 scikit-learn==1.0.2 joblib==1.1.0
```

To be able to follow through: 

Pandas [documentation](https://pandas.pydata.org/docs/)

If you prefer [video](https://www.youtube.com/watch?v=vmEHCJofslg), I got you.

----
## Data Exploration
The Dataset contains records of information gathered from Malware and Not-Malware Portable Executable (PE) Files.

In this section we will do some of the intial exploration of the data (and how it was generated) and begin to create some graphs and summary statistics to understand the dataset better.

Beyond just printing out values of records and looking through them by hand it is important to spend time on a step of the process called Exploratory Data Analysis (EDA) in which we create and analyze graphs, summary statistics and other views of the data to understand it better. This usually also means reading documentation about the data to understand how we should use it.


### Feature set
To learn more about how this dataset was created see the [github page](https://github.com/urwithajit9/ClaMP). The features for this dataset are extracted from a windows PE file using [pefile](https://github.com/erocarrera/pefile). 

You will probably need to do some aditional research outside of what we provide here to understand what a PE file is and what these features relate to.

Some Additional Resources:

- https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
- https://en.wikipedia.org/wiki/Portable_Executable
- https://medium.com/ax1al/a-brief-introduction-to-pe-format-6052914cc8dd
- https://resources.infosecinstitute.com/topic/presenting-the-pe-header/

----
## Section 2: Preparing the Data
Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.

> **NOTE:** If you are interested in learning more about machine learning in general, or need an explanation of some of the concepts covered, a good resource is the Google Machine Learning Crash Course found at https://developers.google.com/machine-learning/crash-course

### Normalizing Numerical Features
In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution however, normalization ensures that each feature is treated equally when applying supervised learners.

We will use [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for this.

### Implementation: Data Preprocessing One Hot Encoding

From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.


### Joining Data
Finally we have our numeric and our categorical data so we are ready to join the columns together to create a final features dataset.

### Train/Test Dataset Split
Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. 
Run the code cell below to perform this split.

---

## Section 3: Creating Models on the Data

Now that the Dataset has been Preprocessed it is time to create Models using it. In our case we would like to know if a given row of the dataset is malware or not malware. This Modeling task is called **Classification**. We can use anything from Naive to Complex Models and in this project will only touch on a few different types of models. There are many additional classification model types but most will fit into similar training/analysis pattern.

### Section 3.1: Supervised Learning Models
**The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **which is the easiest place to start when creating models:**
- Gaussian Naive Bayes (GaussianNB)
- Decision Trees
- Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
- K-Nearest Neighbors (KNeighbors)
- Stochastic Gradient Descent Classifier (SGDC)
- Support Vector Machines (SVM)
- Logistic Regression

> **NOTE:** To learn more about supervised learning models, a useful resource is the scikit-learning documentation at https://scikit-learn.org/stable/supervised_learning.html. 

> If you prefer video lectures, the freeCodeCamp Machine Learning in Python Tutorial at https://www.youtube.com/watch?v=pqNCD_5r0IU may be helpful.

#### **Section 3.1.1: Logistic Regression**

*What is Logistic Regression?*

This type of statistical analysis (also known as logit model) is often used for predictive analytics and modeling, and extends to applications in machine learning. In this analytics approach, the dependent variable is finite or categorical: either A or B (binary regression) or a range of finite options A, B, C or D (multinomial regression). It is used in statistical software to understand the relationship between the dependent variable and one or more independent variables by estimating probabilities using a logistic regression equation. 

This type of analysis can help you predict the likelihood of an event happening or a choice being made. For example, you may want to know the likelihood of a visitor choosing an offer made on your website — or not (dependent variable). Your analysis can look at known characteristics of visitors, such as sites they came from, repeat visits to your site, behavior on your site (independent variables). 


https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

- **Initialize Model**
- **Train Model**
- **Analyze Model Results**

#### **Section 3.1.2: Random Forest**

*What is Random forest?*

The random forest algorithm is an extension of the bagging method as it utilizes both bagging and feature randomness to create an uncorrelated forest of decision trees. Feature randomness, also known as feature bagging or “the random subspace method”, generates a random subset of features, which ensures low correlation among decision trees. This is a key difference between decision trees and random forests. While decision trees consider all the possible feature splits, random forests only select a subset of those features.

By accounting for all the potential variability in the data, we can reduce the risk of overfitting, bias, and overall variance, resulting in more precise predictions.

https://scikit-learn.org/stable/modules/ensemble.html#random-forests

- **Initialize Model**
- **Train Model**
- **Analyze Model Results**

#### **Section 3.1.3: Gradient Boosting**

*What is Gradient Boosting?*

Gradient boosting is a technique used in creating models for prediction. The technique is mostly used in regression and classification procedures. Prediction models are often presented as decision trees for choosing the best prediction. Gradient boosting presents model building in stages, just like other boosting methods, while allowing the generalization and optimization of differentiable loss functions.


- **Initialize Model**
- **Train Model**
- **Analyze Model Results**

### Section 3.2: Unsupervised Machine Learning

The previous 3 sections we had features and targets and were trying to predict the value of something. With unsupervised learning we are just using the features to try to group them in some way or to gather some additional insights without actually predicting malware or not malware

#### **Section 3.2.1: Elbow method k means python**

K-Means is an unsupervised machine learning algorithm that groups data into k number of clusters. The number of clusters is user-defined and the algorithm will try to group the data even if this number is not optimal for the specific case.

Therefore we have to come up with a technique that somehow will help us decide how many clusters we should use for the K-Means model.

The Elbow method is a very popular technique and the idea is to run k-means clustering for a range of clusters k (let’s say from 1 to 10) and for each value, we are calculating the sum of squared distances from each point to its assigned center(distortions).

When the distortions are plotted and the plot looks like an arm then the “elbow”(the point of inflection on the curve) is the best value of k.
    
Use the following Parameters in Kmeans:

kmeans_kwargs = {  
   ...:     "init": "random",  
   ...:     "n_init": 10,  
   ...:     "max_iter": 300,  
   ...:     "random_state": 0,  
   ...: }  
   
   
   
#### **Section 3.2.2: Principal Component Analysis(PCA)**
What is Principal Component Analysis?

PCA is a dimensionality reduction framework in machine learning. According to Wikipedia, PCA (or Principal Component Analysis) is a “statistical procedure that uses orthogonal transformation to convert a set of observations of possibly correlated variables…into a set of values of linearly uncorrelated variables called principal components.”

The Benefits of PCA (Principal Component Analysis)
PCA is an unsupervised learning technique that offers a number of benefits. For example, by reducing the dimensionality of the data, PCA enables us to better generalize machine learning models. This helps us deal with the “curse of dimensionality”.

Most, if not all, algorithm performance depends on the dimension of the data. Models running on very high dimensional data might perform very slow—or even fail—and require significant server resources. PCA can help us improve performance at a very low cost of model accuracy. 

Other benefits of PCA include reduction of noise in the data, feature selection (to a certain extent), and the ability to produce independent, uncorrelated features of the data. PCA also allows us to visualize data and allow for the inspection of clustering/classification algorithms. 

**Directions:** please create a PCA with 5 features and set the randomness with your seed

### Section 3.3: Supervised and Unsupervised Models combined  

Next you will combine the Unsupervised and Supervised Machine Learning to predict the target Malware analysis with each of the two datasets you generated above (in the k-means and PCA sections)

#### **Section 3.3.1: K-means + Gradient Boosting**

Now use the K-means feature dataframe you generated for the k-means Clustering to train and test a Gradient Boosting model. Use the additional cluster feature as training/testing features and the (untouched) labels 

- **Initialize Model**
- **Train Model**
- **Analyze Model Results**

#### **Section 3.3.2: PCA Features + Gradient Boosting**

Now use the PCA feature dataframe you generated for the PCA Clustering to train and test a Gradient Boosting model. Use the PCA features as training/testing features and the (untouched) labels 

**(Make sure you didnt run PCA on the labels(class) and only ran it on the features!!)**

- **Initialize Model**
- **Train Model**
- **Analyze Model Results**



**Happy coding!**
