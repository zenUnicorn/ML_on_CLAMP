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

### Data Exploration
The Dataset contains records of information gathered from Malware and Not-Malware Portable Executable (PE) Files.

In this section we will do some of the intial exploration of the data (and how it was generated) and begin to create some graphs and summary statistics to understand the dataset better.

Beyond just printing out values of records and looking through them by hand it is important to spend time on a step of the process called Exploratory Data Analysis (EDA) in which we create and analyze graphs, summary statistics and other views of the data to understand it better. This usually also means reading documentation about the data to understand how we should use it.


#### Feature set
To learn more about how this dataset was created see the [github page](https://github.com/urwithajit9/ClaMP). The features for this dataset are extracted from a windows PE file using [pefile](https://github.com/erocarrera/pefile). 

You will probably need to do some aditional research outside of what we provide here to understand what a PE file is and what these features relate to.

Some Additional Resources:

- https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
- https://en.wikipedia.org/wiki/Portable_Executable
- https://medium.com/ax1al/a-brief-introduction-to-pe-format-6052914cc8dd
- https://resources.infosecinstitute.com/topic/presenting-the-pe-header/



Happy coding!
