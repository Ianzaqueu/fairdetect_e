
# Welcome to *Exploration and identification of Bias in Synthetic Card Approval Dataset * 
***by Group E***

![enter image description here](https://i.postimg.cc/s2drfjk9/Whats-App-Image-2022-07-31-at-12-51-37-PM.jpg)

  ## Table of Contents 


|Number of order                          |Title 
|----------------|-------------------------------|
|1 |`Description of project`|
|2 |`Requirements` |
|3 |`Group E's package`|
|4 |`Installation`|
|5 |`Project Flow` |
|6 |`Status`|
|7 |`Collaborators`|
  

## Description of project

  

The scope of the project is centred around the exploration of any potential bias in the Synthetic Card Approval Dataset, and the engagement of modelling using _FairDetect_ by Ryan Daher, a library developed by Group E and IBM’s AIF360. As Group E, our project seeks to identify potential biases in the rate of credit card approval given to customer within the dataset, due to a natural characteristic which places a group of customers at an unfair disadvantage in comparison to another group of customers within the same institution.


Ryan Daher, a former IE University student, engaged in the exploration of the case study in his final dissertation entitled _‘Transparent Unfairness: An Approach to Investigating Machine Learning Bias’_ - assuming using the same dataset. Within his project scope emerged the creation of a library entitled FairDetect.py, which contains tools for data analysis, model building and data exploration. It is on the basis of Ryan Daher’s package, that our group E has created a package which will focuses on better usability, increased speed and reliability of using the code through Jupyter Notebook in the exploration of bias. Finally, the use of IBM’s AIF360 - open source toolkit that helps examine, report, and mitigate discrimination and bias in machine learning models throughout the AI application lifecycle - is focused on ensuring that critical analysis of Ryan Daher’s ML model is achieved in order to ensure that the correct bias is identified and subsequently mitigated. The project shall implement the use of AIF360 in order to engage in a comparative analysis with Ryan Daher’s application of the FairDetect package, and ensure that results are following suit.

  
  

## Requirements


  

-   Matplotlib.pyplot
-   Numpy
-   Pandas
-   Seaborn
-   Image from IPython.Display

  

## Inside Group E’s package:

  

-   Train_test_split from sklearn.model_selection
-   Display from IPython.display
-   Declarative_base from sqlalchemy.ext.declarative
-   Column, Integer, String, DateTime, Float from sqlalchemy
-   Randrange from random
-   Make_subplots from plotly.subplots
-   Plotly.graph_objects
-   Dalex
-   Chi2_contingency from scipy.stats
-   Tabulate from tabulate
-   Confusion_matrix from sklearn.metrics
-   Chisquare from scipy.stats
-   Precision_score from sklearn.metrics
-   Accuracy_score from sklearn.metrics
-   MLPClassifier from sklearn.neural_network

  

## Installation:

-   AIF360 Tool kit (pip install)
-   FairDetect (pip install)
-   Group E library (pip install)

  

## Project Flow:

  

The flow of configuration and implementation is showcased in the jupyter notebook provided in the repository.

  

-   Begin with a pip installation of the libraries; FairDetect, Group E Library and the AIF360.
-   A clear EDA of the dataset is conducted

  

Begin by implementing Ryan Daher’s FairDetect model, with the use of Group E’s library.

 1.   Proceed with a preprocessing and splitting of the dataset by invoking functions already in Group E’s library
 2.   Proceed to train the ML Model prebuilt models such as 'LOG' (logistic regression), 'XGB' (xgboost), 'MLP' (artificial neural network), 'RFC' (random forest classifier), 'DTC' (decision tree classifier).
 3.   Invoke the ‘identify_bias’ methodology
 4.   Analyse results of the prebuilt models
 5.   Invoke Understand_SHAP method which lays within our extended dataframe class

  

Implement a similar flow using IBM’s AIF360

 1.   Upon a pip installation of the AIF360, prepare the full data set with original response, the train dataset with the original response, the test set with the original response, the training set with the predictive response, and the test set with predictive response
 2.   Engage in a quick EDA
 3.   Using the BinaryLabelDatasetMetric, explore the methodology regarding the dataset.
 4.   Implement the ClassificationMetric by creating two BinaryLabelDataset objects; real outcome vs predicted outcome

  

>Finalise by engaging in a critical comparative analysis of both approaches concerning bias within the model, and within the dataset.

  

## Status:

  

Project has been completed successfully, in pair with a full package which allows users for a complete exploration and reproduction of the project outputs. The project has been conducted over a period of 4 weeks using Ryan Daher’s fairdect library and IBM’s AIF360 library. A conclusion has been established upon further comparative analysis; although accepting further improvements and bias detection consideration points.

  

## Collaborators:

  

-   Dareen Shaheen Hamdy: https://github.com/DareenShaheen
-   Amanda Moles de Matos: https://github.com/amandamolesm
-   Javier Nieves Remacha: [https://github.com/jnremachaGH](https://github.com/jnremachaGH)
-   Mohamad Rizk: [https://github.com/MohamadRizk](https://github.com/MohamadRizk)
-   Ismael Ayat: [https://github.com/Ismmael](https://github.com/Ismmael)
-   Ian Zaqueu: [https://github.com/Ianzaqueu](https://github.com/Ianzaqueu)





