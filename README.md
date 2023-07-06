# CKD Predictor ML Analysis  
A machine learning project using various algorithms including neural networks, clustering, decision trees, and naive Bayesian to predict the presence of Chronic Kidney Disease (CKD) in patients based on a dataset of 400 samples. The project also showcases a preprocessing and data cleaning process. 
The project is part of OpenU Data-mining course, the dataset was given.
  
Part of OpenU Data-mining course final project.  
  
Please note, this project is for educational purposes and the dataset size is relatively small for real-world applications.  
    
## Structure  
  
- ```main.py``` : The main driver script that invokes functions from other modules. This is where you should run the project, _Important_ - see 'Usage' section.  
  
- ```helper.py``` : Contains functions for loading the dataset, preprocessing and cleaning data, as well as data exploration and visualization.  
  
- ```models.py``` : Includes Decision Tree / Naive Bayes / Clustering models training and evaluation functions  
  
- ```neural_network.py``` : Handles building, training, evaluating, and plotting of the neural network model.  
  
## Requirements  
  
- numpy  
- pandas  
- sklearn  
- scipy  
- matplotlib  
- seaborn  
- tensorflow  
  
_Use ```pip install -r requirements.txt``` to auto install_  
  
## Dataset Variations  
  
In the ```datasetVariations``` directory you would find many variations of the Chronic Kidney Disease (CKD) Dataset, the variations are steps taken during the project, for example, ```CKDF_filledAllNumeric``` is the CKD dataset variation which is already filled the missing values and turn all the data into numeric form.  
  
In ```main.py```, the process is starting with ```CKDF_noQmarks_unindexed```, which is the CKD dataset variation which is replaced all the Question marks which were (because originally it was ```.arff``` format) and replaced by ```.csv``` appropriate format, also, removed the _index_ column.
  
## Workflow  
  
The main workflow is sticking to best practices;  
  
- Load the data.  
- EDA (not implemented in ```main.py```).  
- Split into training/validation/test sets.  
- Data Preprocessing (validation and test sets are preprocessed with training set parameters).  
- Training  
- Evaluation  
  
## Usage  
  
You should run ```python main.py```  
  
```main.py``` is divided into three Sections, _ Each Section must be running seperately from the other ones, thus when you decide which section to run, you must comment the other sections, else unwanted behaviour will occur.  
  
In the current state of the project, Section 1 is ready for running, Sections 2,3 are in comment.  
  
- Section 1 : Supervised ML (not deep learning) models (no validation set).    
- Section 2 : Unsupervised clustering models (only kmeans for now), no split.  
- Section 3 : Predefined simple and basic feedforward Neural Network model (with validation set).
  
## PDF's  
  
As described in the starting description, this was part of a Data-mining course, which was splitted into two parts.  
  
The PDF's are the project in-depth analysis, they're written in Hebrew.  
  
If you really want an English version, contact me, I'll translate it.  
  
## Results  
  
I'll give you the honor to run and check the results, there are some ```.png```'s of the results I got in the ```resultsPngs``` directory.  

  
