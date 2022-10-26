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

|   | someFeature |                    | someFeature_A | someFeature_B | someFeature_C |
| :-: | :-: |                            | :-: | :-: | :-: |
| 0 |  B  |  | 0 | 1 | 0 |
| 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |
| 2 |  A  |  | 1 | 0 | 0 |


### Joining Data
Finally we have our numeric and our categorical data so we are ready to join the columns together to create a final features dataset.

### Train/Test Dataset Split
Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. 
Run the code cell below to perform this split.

---

## Section 3: Creating Models on the Data

Now that the Dataset has been Preprocessed it is time to create Models using it. In our case we would like to know if a given row of the dataset is malware or not malware. This Modeling task is called **Classification**. We can use anything from Naive to Complex Models and in this project will only touch on a few different types of models. There are many additional classification model types but most will fit into similar training/analysis pattern.



Happy coding!
