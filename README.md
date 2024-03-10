# AI Domestic Violence Support System
![Banner Image](https://github.com/keerthigavnr/Intel-AI-oneAPI-Hackathon/blob/master/image.png)

The AI Domestic Violence Support System utilizes advanced NLP and machine learning models to identify various types of domestic abuse from user inputs. Designed with privacy and user-friendliness at its core, our system provides personalized insights and resources to help users understand their situations better.

## Table of Contents

- [About the Project](#about-the-project)
- [Social Impact](#social-impact)
- [Tech stack](#tech-stack)
- [Use of oneAPI in our project](#using-oneapi)
- [Working](#working)
- [References for Datasets](#references-for-datasets)

## About the Project
Our project is designed to listen and understand, taking user inputs in the form of text descriptions about their feelings and the situations they're experiencing. At its core, it harnesses powerful NLP techniques to process these inputs, identifying the type of abuse and the context of the user's situation. By analyzing the nuances of the language and the details shared, our system can discern the specific nature of the abuse. This sophisticated understanding allows it to provide targeted knowledge and resources that are tailored to the user's individual circumstances.

## Social Impact

Our project helps people who have faced harm at home by giving them information to understand and heal. When someone tells the system about their tough situations, it uses special computer techniques to figure out the kind of harm they're facing. Then, it gives them advice and help that fits their situation. Our project makes a big positive change. It helps people feel less alone and more supported. It shows them that there are ways to get better and that people care about helping them. This can change their lives in a good way, by making them feel safer and more hopeful.

## Tech stack

- Intel® OneAPI
- NLP
- Catboost classifier
- Flask
- HTML,Bootstrap

## Using oneAPI

### Accelerating Machine Learning with Intel® Extension for Scikit-learn

```python
!pip install scikit-learn-intelex
```
Activate the extension to accelerate Scikit-learn algorithms:

```python
from sklearnex import patch_sklearn
patch_sklearn()
```
With the environment set up and the Intel® extension activated, proceeding to run the PCA and Logistic Regression models which are accelerated for improved performance.

PCA:

```python
from sklearnex.decomposition import PCA
pca = PCA(n_components=30)
train_pca = pca.fit_transform(train_embeddings)
test_pca = pca.transform(test_embeddings)
```
Logistic Regression:

```python
from sklearnex.linear_model import LogisticRegression
log_reg_sklearnex = make_pipeline(TfidfVectorizer(), LogisticRegression(C=2.0, max_iter=500))
```

### Performance analysis

![Logistic regression](https://github.com/keerthigavnr/Intel-AI-oneAPI-Hackathon/blob/master/images/performance1.png)

Performance comparision for LOGISTIC REGRESSION with and without oneDAl

![PCA](https://github.com/keerthigavnr/Intel-AI-oneAPI-Hackathon/blob/master/images/performance2.png)

Performance comparision with PCA and without oneDAl

## Working

![Input](https://github.com/keerthigavnr/Intel-AI-oneAPI-Hackathon/blob/master/images/working1.png)

![Content](https://github.com/keerthigavnr/Intel-AI-oneAPI-Hackathon/blob/master/images/working2.png)

## References for Datasets

- [Dataset source](https://huggingface.co/datasets/Spiderman01/Domestic_violence_info_support_fromposts)


