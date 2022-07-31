

import pandas as pd
import numpy as np
########################################################################################

from sklearn.model_selection import train_test_split
from IPython.display import display
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy
########################################################################################
import matplotlib.pyplot as plt
from random import randrange
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from scipy.stats import chisquare
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
#from os.path import exists
#######################################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost
from sklearn.tree import DecisionTreeClassifier
import shap

#######################################################################################

Base = declarative_base()

#######################################################################################


class FairDetectE(pd.DataFrame):
    
    def __init__(self, pd_DF, data_filename = None, target_var=None, ml_model_object=None, 
                 ml_prebuilt_model=None, *args, **kwargs):
        """
        Method that initializes an object of the class fairdetect_group_e with the 
        given arguments.

        Parameters
        ----------
        pd_DF : pandas.DataFrame
            Input DataFrame with all the data that will be needed for the bias analysis.
        data_filename : str, optional
            User input name of pandas dataframe, to be stored in persistent
            sqlite3 database.The default is None.
        target_var : str, optional
            Name of the target variable. The default is None.
        ml_model_object : sklearn or xgboost, optional
            model object from sklearn or xgboost. The default is None.
        ml_prebuilt_model : str, optional
            Acronym of the ML algorithm that will be used to generate the prediction.
            Options: 'DTC', 'XGB', 'MLP', 'RFC', 'LOG'.The default is None.

        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        Exception
            When the target_var is not present in the DataFrame.
            Or, when ml_prebuilt_model is not one of the supported ones:
                'DTC', 'XGB', 'MLP', 'RFC', 'LOG'

        Returns
        -------
        None.

        """
        
        super(FairDetectE, self).__init__(pd_DF, *args, **kwargs)

        if pd_DF is not None:
            try:
                pd.DataFrame(pd_DF)
            except:
                return(print("Type error: Variable pd_DF is of incorrect type"))

        if data_filename is not None:
            try:
                if not isinstance(data_filename, str):
                    raise TypeError
            except TypeError:
                return(print("Type error: Variable data_filename should be of type str"))
        
        if target_var is not None:
            try:
                if not isinstance(target_var, str):
                    raise TypeError
            except TypeError:
                return(print("Type error: Variable target_var should be of type str"))

        if ml_prebuilt_model is not None:
            try:
                if not isinstance(ml_prebuilt_model, str):
                    raise TypeError
            except TypeError:
                return(print("Type error: Variable ml_prebuilt_model should be of type str"))
        
        if ml_prebuilt_model is not None:
            try:
                if ml_prebuilt_model not in ['DTC', 'XGB', 'MLP', 'RFC', 'LOG']:
                    raise Exception
            except Exception:
                return(print("Exception: ml_prebuilt should be: 'DTC', 'XGB', \
                         'MLP', 'RFC' or 'LOG' "))
        
        # Checking that pd_DF contains column "target_var"
        if target_var is not None:
            try: 
                found_target_var = False
                for (columnName, columnData) in pd_DF.iteritems():
                    if columnName == target_var:
                        found_target_var = True
                if not found_target_var:
                    raise Exception
            except Exception:
                return(print("There is no column in the DataFrame named " + str(target_var)))
        
        
        self.pd_DF = pd_DF
        self.target_var = target_var
        self.ml_model_object = ml_model_object
        self.ml_prebuilt_model = ml_prebuilt_model
        self.data_filename = data_filename
        self.results_stored = pd.DataFrame(columns=['filename','target_var','analysis_date','method',
                                                    'ml_model','metric','result','comments'])
        FairDetectE.data_filename, FairDetectE.target_var, \
        FairDetectE.ml_model_object, FairDetectE.ml_prebuilt_model = None, None, None, None
    
    @property
    def _constructor(self):
        def func_(*args,**kwargs):
            df = FairDetectE(*args,**kwargs)
            return df
        return func_
    

    @classmethod
    def from_file(cls, filename, target_var=None, ml_model_object=None, ml_prebuilt_model=None):
        """
        Converts the input file into a DataFrame and calls the class __init__ method with it.

        Parameters
        ----------
        filename : str
            Name of the input file, must be one of the following types: csv, xls, xlsx, xml or json.
        target_var : str, optional
            Name of the target variable. The default is None.
        ml_model_object : sklearn or xgboost, optional
            model object from sklearn or xgboost. The default is None.
        ml_prebuilt_model : str, optional
            Acronym of the ML algorithm that will be used to generate the prediction.
            Options: 'DTC', 'XGB', 'MLP', 'RFC', 'LOG'.The default is None.

        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        Exception
            When the target_var is not present in the DataFrame.
            Or, when ml_prebuilt_model is not one of the supported ones:
                'DTC', 'XGB', 'MLP', 'RFC', 'LOG'

        Returns
        -------
        Required information for __init__ to initialize instance of 
        class fairdetect_group_e

        """
        try:
            if not isinstance(filename, str):
                raise TypeError
        except TypeError:
            return(print("The filename must be of type str"))
        
        try:
            if not filename.endswith((".csv",".xls",".xlsx",".xml",".json")):
                raise Exception
        except Exception:
            return(print("The file must be of the following types: csv, xls, xlsx, xml or json"))

        if target_var is not None:
            try:
                if not isinstance(target_var, str):
                    raise TypeError
            except TypeError:
                    return(print("Type error: Variable target_var should be of type str"))
    
        if ml_prebuilt_model is not None:
            try:
                if not isinstance(ml_prebuilt_model, str):
                    raise TypeError
            except TypeError:
                return(print("Type error: Variable ml_prebuilt_model should be of type str"))
        
        if ml_prebuilt_model is not None:
            try:
                if ml_prebuilt_model not in ['DTC', 'XGB', 'MLP', 'RFC', 'LOG']:
                    raise Exception
            except Exception:
                return(print("Exception: ml_prebuilt should be: 'DTC', 'XGB', \
                         'MLP', 'RFC' or 'LOG' "))
        
        if filename.endswith(".csv"):
            pd_import = pd.read_csv(filename)
            cls.target_var = target_var
            cls.ml_model_object = ml_model_object
            cls.ml_prebuilt_model = ml_prebuilt_model
            cls.data_filename = filename
            return cls(pd_import, cls.data_filename, cls.target_var, cls.ml_model_object, cls.ml_prebuilt_model)
        elif filename.endswith(".xls"):
            pd_import = pd.read_excel(filename)
            cls.target_var = target_var
            cls.ml_model_object = ml_model_object
            cls.ml_prebuilt_model = ml_prebuilt_model
            cls.data_filename = filename
            return cls(pd_import, cls.data_filename, cls.target_var, cls.ml_model_object, cls.ml_prebuilt_model)
        elif filename.endswith(".xlsx"):
            pd_import = pd.read_excel(filename)
            cls.target_var = target_var
            cls.ml_model_object = ml_model_object
            cls.ml_prebuilt_model = ml_prebuilt_model
            cls.data_filename = filename
            return cls(pd_import, cls.data_filename, cls.target_var, cls.ml_model_object, cls.ml_prebuilt_model)
        elif filename.endswith(".xml"):
            pd_import = pd.read_xml(filename)
            cls.target_var = target_var
            cls.ml_model_object = ml_model_object
            cls.ml_prebuilt_model = ml_prebuilt_model
            cls.data_filename = filename
            return cls(pd_import, cls.data_filename, cls.target_var, cls.ml_model_object, cls.ml_prebuilt_model)
        elif filename.endswith(".json"):
            pd_import = pd.read_json(filename)
            cls.target_var = target_var
            cls.ml_model_object = ml_model_object
            cls.ml_prebuilt_model = ml_prebuilt_model
            cls.data_filename = filename
            return cls(pd_import, cls.data_filename, cls.target_var, cls.ml_model_object, cls.ml_prebuilt_model)
    
    
    
    ####################################################################################################
    
    # Intermediate useful methods for data treatment

    def encode_labels(self, df=None):
        """

        Obtains features / attributes of object type from a dataset and encodes them into integers. When call it,
        a dataframe must be passed to it. 

        Parameters
        ----------
        df : pandas dataframe
            A pandas dataframe which has attributes of "object" type features to be encoded
            
        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        ValueError
            When the dataframe is empty.     
           
        Returns
        -------
        TYPE pandas dataframe
            The function returns a pandas dataframe with all features of type integers or float.

        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable df should be of type pd.DataFrame."))
        try:
            if df.empty:
                raise ValueError
        except ValueError:
            return(print("Value error: Variable df is empty."))
 
        
        if df!=None: self.pd_DF = df
        le = preprocessing.LabelEncoder()
        objList = df.select_dtypes(include = "object").columns
        for attr in objList: #attr is the featur or columns
            self.pd_DF[attr] = le.fit_transform(self.pd_DF[attr])
        return self.pd_DF

        
    def X_y_split(self, target_var = None):
        """
        
        Split the given dataframe into two:
            y_data that has the "target" feature only,
            &
            X_data gat has all the features except the "target" feature.
            The two new dataframes have same number of rows.

        Parameters
        ----------
        target_var: str
            the "target" feature that will be used to split the given dataframe.

        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        ValueError
            When the variable, the feature name, is empty.     

        Returns
        -------
        pandas dataframes
            The function returns two datasets, one containing the target variable
            and one containing all the remaining features.

        """
        if target_var is not None:
            try:
                if not isinstance(target_var, str):
                    raise TypeError
            except TypeError:
                return(print("Type error: Variable target_var should be of type string. It is the name of the target feature"))
            try:
                if not target_var:
                    raise ValueError
            except ValueError:
                return(print("Value error: Variable target_var is empty"))      
            try:
                if not target_var in list(self.pd_DF.columns):
                    raise ValueError
            except ValueError:
                return(print("Value error: Variable target_var does not exist in the dataframe"))
        
        
        if target_var is not None: self.target_var = target_var
        self.y_data = self.pd_DF[[self.target_var]].copy()
        self.X_data = self.pd_DF.drop([self.target_var], axis=1).copy()
        print("Succesfully splitted: showing first 3 rows...")
        display(self.y_data.head(3)),display(self.X_data.head(3)) 

    def split_data_totrain(self, train_size_split = 0.8, random_state_split=0):
        """
    
        Split the given dataframe into two, based on a given split ratio:
        1 dataframe to train the model and 1 to test the model.

        Parameters
        ----------
        train_size_split: float
            The split ratio used to determine the size of each split for training and testing dataframes.
        random_state_split: int
            Ensures that the generated splits are reproducible.

        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        ValueError
            When the variable, the feature name, is empty.     

        Returns
        -------
        pandas dataframes
            The function returns 4 dataframes, two for training and two for testing.
            First split contains training dataframes:
                X_train: contains all features except the target feature.
                y_train: contains target feature.
            Second split contains test dataframes:
                X_test: contains all features except the target feature.
                y_test: contains target feature.

        """
        
        try:
            if not isinstance(train_size_split, float):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable train_size_split should be of type float."))
        try:
            if not 0< train_size_split <1:
                raise ValueError
        except ValueError:
            return(print("Value error: Variable train_size_split should be netween 0 and 1."))
        try:
            if not isinstance(random_state_split, int):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable random_state_split should be of type int."))
        try:
            if random_state_split < 0:
                raise ValueError
        except ValueError:
            return(print("Value error: Variable random_state_split should be postive int."))

        
        self.random_state_split = random_state_split
        self.train_size_split = train_size_split
        self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(self.X_data, self.y_data, train_size = \
                                 self.train_size_split, random_state = self.random_state_split)
        print("Succesfully splitted: showing first 3 rows of Train...")
        display(self.X_train.head(3)), display(self.y_train.head(3))
        print("Succesfully splitted: showing first 3 rows of Test...")
        display(self.X_test.head(3)), display(self.y_test.head(3))
        return self.X_train, self.X_test, self.y_train, self.y_test 


    #######################################################################################

    # Machine Learning Built-in Methods

    def train_model(self, ml_prebuilt_model = None, ml_model_object=None):
        """
        This method takes the user input machine learning model desired, and 
        trains it with the X_train dataset of the instance. 
        The user can provide prebuilt models ('DTC', 'XGB', 'MLP', 'RFC', 'LOG') 
        or pass a model object.

        Parameters
        ----------
        ml_prebuilt_model : str, optional
            Acronym of the ML algorithm that will be used to generate the prediction.
            Options: 'DTC', 'XGB', 'MLP', 'RFC', 'LOG'.The default is None.
        ml_model_object : sklearn or xgboost, optional
            model object from sklearn or xgboost. The default is None.


        Returns
        -------
        None.

        """
                
        if ml_prebuilt_model is not None:
            try:
                if ml_prebuilt_model not in ['DTC', 'XGB', 'MLP', 'RFC', 'LOG']:
                    raise Exception
            except Exception:
                return(print("Exception: ml_prebuilt should be: 'DTC', 'XGB', \
                         'MLP', 'RFC' or 'LOG' "))


        
        if (ml_model_object is not None) or ((ml_prebuilt_model is None)&(ml_model_object is None)&(self.ml_model_object is not None)):
            print("User input Model Object Selected")
            self.ml_prebuilt_model = None
            if ml_model_object is not None: self.ml_model_object = ml_model_object
            if ml_model_object is not None: self.ml_model_desc = ml_model_object
            self.ml_model_desc = self.ml_model_object
            self.model = self.ml_model_object.fit(self.X_train, self.y_train)
            self.results_stored.loc[-1] = [self.data_filename,self.target_var,datetime.now(), 'train_model', str(self.ml_model_desc), \
                                        'Training Accuracy',accuracy_score(self.y_train, self.model.predict(self.X_train)),\
                                        "User input Model Object Selected"]
            self.results_stored.index = self.results_stored.index + 1  # shifting index
            self.results_stored.sort_index(inplace=True) 
            self.results_stored.loc[-1] = [self.data_filename,self.target_var,datetime.now(), 'train_model', str(self.ml_model_desc), \
                                        'Testing Accuracy',accuracy_score(self.y_test, self.model.predict(self.X_test)),\
                                        "User input Model Object Selected"]
            self.results_stored.index = self.results_stored.index + 1  # shifting index
            self.results_stored.sort_index(inplace=True)            
            print("Training Classification Accuracy:", accuracy_score(self.y_train, self.model.predict(self.X_train)))
            print("Testing Classification Accuracy:", accuracy_score(self.y_test, self.model.predict(self.X_test)))
            print("User input Model Succesfully fitted!")

        elif (ml_model_object is None) & (ml_prebuilt_model is not None):
            self.ml_model_object = None
            self.ml_prebuilt_model = ml_prebuilt_model

            if self.ml_prebuilt_model == "LOG":
                print("Logistic Regression selected")
                '''exceptions'''
                self.ml_model_desc = LogisticRegression()
                self.model = LogisticRegression().fit(self.X_train, self.y_train)            
                print("Training Classification Accuracy:", accuracy_score(self.y_train, self.model.predict(self.X_train)))
                print("Testing Classification Accuracy:", accuracy_score(self.y_test, self.model.predict(self.X_test)))
                print("Logistic Regression Model Succesfully fitted!")
            if self.ml_prebuilt_model == "XGB":  
                print("XGB selected")
                '''exceptions'''
                self.ml_model_desc = xgboost.XGBClassifier()
                self.model = xgboost.XGBClassifier().fit(self.X_train, self.y_train)
                print("Training Classification Accuracy:", accuracy_score(self.y_train, self.model.predict(self.X_train)))
                print("Testing Classification Accuracy:", accuracy_score(self.y_test, self.model.predict(self.X_test)))          
                print("XGBoost Model Succesfully fitted!")
            if self.ml_prebuilt_model == "MLP":  
                print("Artificial Neural Network Classifier selected (MLP Classifier)")
                '''exceptions'''
                self.ml_model_desc = MLPClassifier()
                self.model = MLPClassifier().fit(self.X_train, np.array(self.y_train).reshape(-1,1))
                print("Training Classification Accuracy:", accuracy_score(np.array(self.y_train).reshape(-1,1), self.model.predict(self.X_train)))
                print("Testing Classification Accuracy:", accuracy_score(np.array(self.y_test).reshape(-1,1), self.model.predict(self.X_test)))            
                print("Artificial Neural Network Model Succesfully fitted!")
            if self.ml_prebuilt_model == "RFC":  
                print("Random Forest Classifier selected")
                '''exceptions'''
                self.ml_model_desc = RandomForestClassifier()
                self.model = RandomForestClassifier().fit(self.X_train, self.y_train)
                print("Training Classification Accuracy:", accuracy_score(self.y_train, self.model.predict(self.X_train)))
                print("Testing Classification Accuracy:", accuracy_score(self.y_test, self.model.predict(self.X_test)))            
                print("Random Forest Classifier Model Succesfully fitted!")
            if self.ml_prebuilt_model == "DTC":  
                print("Decision TreeClassifier selected")
                '''exceptions'''
                self.ml_model_desc = DecisionTreeClassifier()
                self.model = DecisionTreeClassifier().fit(self.X_train, self.y_train)
                print("Training Classification Accuracy:", accuracy_score(self.y_train, self.model.predict(self.X_train)))
                print("Testing Classification Accuracy:", accuracy_score(self.y_test, self.model.predict(self.X_test)))
                print("Decision TreeClassifier Succesfully fitted!")
    
            # Storing results in staging intermediate area results_stored dataframe
            self.results_stored.loc[-1] = [self.data_filename,self.target_var,datetime.now(), 'train_model', str(self.ml_model_desc),
                                               'Training Accuracy',accuracy_score(self.y_train, self.model.predict(self.X_train)),
                                              "ML prebuilt model selected"]
            self.results_stored.index = self.results_stored.index + 1  # shifting index
            self.results_stored.sort_index(inplace=True) 
            self.results_stored.loc[-1] = [self.data_filename,self.target_var,datetime.now(), 'train_model', str(self.ml_model_desc),
                                               'Testing Accuracy',accuracy_score(self.y_test, self.model.predict(self.X_test)),
                                              "ML prebuilt model selected"]
            self.results_stored.index = self.results_stored.index + 1  # shifting index
            self.results_stored.sort_index(inplace=True)
            

    #######################################################################################
    # Functions to store and read to and from sqlite database using sqlalchemy
    
    def store_results_in_db(self):
        """
        
        By calling this method, all the results obtained and automatically saved internally in the 
        temporary pd.DataFrame "results_stored" are saved into "FairDetectDataBase", table 
        "fairdetect_results". After saving into database in a persistent manner, the temporary 
        pd.DataFrame "results_stored" is reinitialized to an empty pandas DataFrame again.
        
        
        Parameters
        ----------
        No user input parameters.
            
        Raises
        ------
        None
        
        Returns
        -------
        None
        
        """
        
        engine = sqlalchemy.create_engine('sqlite:///FairDetectDataBase.db')
        Base.metadata.create_all(engine)
        self.results_stored.to_sql('fairdetect_results',engine, if_exists='append', index=False)
        print("Stored Succesfully: ", self.results_stored.shape[0], " results")
        print("Total of ", pd.read_sql("SELECT * FROM fairdetect_results", engine).shape[0], " records stored in database")
        self.results_stored = pd.DataFrame(columns=['filename','target_var','analysis_date','method',
                                                    'ml_model','metric','result','comments'])

    def read_results_from_db(self, sqlquery=None):
        """

        Reads the results from the database. If no sqlquery is passed, a default query showing the 
        latest 10 results is shown. If a sqlquery is passed, the result from the query is returned as 
        a pandas dataframe.

        Parameters
        ----------
        sqlquery : str
            A string containing a correct SQL query
            
        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
           
        Returns
        -------
        TYPE pandas dataframe
            The function returns a pandas dataframe with all features of type integers or float.

        """
        if sqlquery is not None:
            try:
                if not isinstance(sqlquery, str):
                    raise TypeError
            except TypeError:
                return(print("Type error: sqlquery df should be of type string"))

        
        engine = sqlalchemy.create_engine('sqlite:///FairDetectDataBase.db')
        if sqlquery == None: sqlquery = "SELECT * FROM fairdetect_results ORDER BY analysis_date DESC LIMIT 10"
        return pd.read_sql(sqlquery, engine)
    
    
    #######################################################################################
    # Original fair detect functions
    
    def create_labels(self, X_test, sensitive):
        """
        
        Creates labels for the variable of which the user wants to detect bias

        Parameters
        ----------
        X_test : pandas.DataFrame
            DataFrame with the test data to be analysed.
        sensitive : str
            Name of the sensitive variable from which the labels will be 
            generated.
            
        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        Exception
            When the sensitive label is not present in the DataFrame.

        Returns
        -------        
        sensitive_label : dict
            Dictionary with the sensitive labels
        
        """

        try:
            if not isinstance(X_test, pd.DataFrame):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable X_test should be of type pd.DataFrame"))
             
        try:
            if not isinstance(sensitive, str):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable sensitive should be of type str"))
        
        # Checking that X_test contains column "sensitive"
        try: 
            found_sensitive = False
            for (columnName, columnData) in X_test.iteritems():
                if columnName == sensitive:
                    found_sensitive = True
            if not found_sensitive:
                raise Exception
        except Exception:
            return(print("There is no column in the DataFrame named " + str(sensitive)))

        
        sensitive_label = {}
        
        for i in set(X_test[sensitive]):
            text = "Please Enter Label for Group" +" "+ str(i)+": "
            label = input(text)
            self.sensitive_label[i]=label
        return(sensitive_label)


    def representation(self,X_test,y_test,sensitive,labels,predictions):
        """
        
        Internal method that returns cont_t, sens_df, fig and p, necessary for the 
        identify_bias method. It provides information regarding how the subgroups 
        and response labels are represented in the dataset.

        Parameters
        ----------
        X_test : pandas.DataFrame
            DataFrame containing the independent variables.
        y_test : pandas.DataFrame
            DataFrame containing the dependent variables.
        sensitive : str
            Sensitive variable.
        labels : dict
            dictionnary with all labels.
        predictions : np.array or pandas.DataFrame
            DataFrame containing the model predictions for the response variable.

        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        Exception
            When the sensitive label is not present in the DataFrame.
        ValueError
            When the dict of labels is empty.

        Returns
        -------
        cont_table : tabulate(pandas.DataFrame)
            Table that shows the proportions of positive and negative
            outcomes for each of the subgroups in the sensitive variable.
        sens_df : dict
            Dictionary containing a DataFrame for each subgroup of sensitive variables
        fig : plotly.graph_objects
            Plotly subplots representing :
                - proportion of the different subgroups in the dataset
                - proportion of the different response labels in the dataset
        p : float
            the p-value of the chi square statistical test applied to the contingency 
            table.

        """
        
        try:
            if not isinstance(X_test, pd.DataFrame):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable X_test should be of type pd.DataFrame"))
      
        try:
            if not isinstance(y_test, pd.DataFrame):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable y_test should be of type pd.DataFrame"))        
      
        try:
            if not isinstance(sensitive, str):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable sensitive should be of type str"))
        
        try:
            if not isinstance(labels, dict):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable labels should be of type dict"))
 
        try:
            if not isinstance(predictions, (pd.DataFrame, np.ndarray)):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable predictions should be of type pd.DataFrame or np.ndarray"))
 
        # Checking that X_test contains column "sensitive"
        try:
            found_sensitive = False
            for (columnName, columnData) in X_test.iteritems():
                if columnName == sensitive:
                    found_sensitive = True
            if not found_sensitive:
                raise Exception
        except Exception:
            return(print("There is no column in the DataFrame named " + str(sensitive)))
        
        # Check if argument labels is empty
        try:
            if not bool(labels):
                raise ValueError
        except ValueError:
            return(print("Value error: Variable labels is empty")) 

        full_table = X_test.copy()
        sens_df = {}

        for i in labels:
            full_table['p'] = predictions
            full_table['t'] = y_test
            sens_df[labels[i]] = full_table[full_table[sensitive]==i]

        contigency_p = pd.crosstab(full_table[sensitive], full_table['t']) 
        cp, pp, dofp, expectedp = chi2_contingency(contigency_p) 
        contigency_pct_p = pd.crosstab(full_table[sensitive], full_table['t'], normalize='index')

        sens_rep = {}
        for i in labels:
            sens_rep[labels[i]] = (X_test[sensitive].value_counts()/X_test[sensitive].value_counts().sum())[i]

        labl_rep = {}
        for i in labels:
            labl_rep[str(i)] = (y_test.value_counts()/y_test.value_counts().sum())[i]


        fig = make_subplots(rows=1, cols=2)

        for i in labels:
            fig.add_trace(go.Bar(
            showlegend=False,
            x = [labels[i]],
            y= [sens_rep[labels[i]]]),row=1,col=1)

            fig.add_trace(go.Bar(
            showlegend=False,
            x = [str(i)],
            y= [labl_rep[str(i)]],
            marker_color=['orange','blue'][i]),row=1,col=2)

        c, p, dof, expected = chi2_contingency(contigency_p)
        cont_table = (tabulate(contigency_pct_p.T, headers=labels.values(), tablefmt='fancy_grid'))
        
        return cont_table, sens_df, fig, p



    def ability(self,sens_df,labels):
        """
        
        Calculates the true positive rate, false positive rate, true negative rate
        and false negative rate for a given dictionary of labels and dataframe.
        
        Parameters
        ----------
        sens_df : dict
            Dictionary containing a DataFrame for each subgroup of sensitive variables.
        labels : dict
            Dictionary containing the labels of the sensitive variables.
            
        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        ValueError
            When the dict of labels is empty.

        Returns
        -------
        true_positive_rate : dict
            dictionary containing the true positive rates
        false_positive_rate : dict
            dictionary containing the false positive rates
        true_negative_rate : dict
            dictionary containing the true negative rates
        false_negative_rate : dict
            dictionary containing the false negative rates

        """
        
        try:
            if not isinstance(sens_df, dict):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable sens_df should be of type dict"))
      
        try:
            if not isinstance(labels, dict):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable labels should be of type dict"))
        
        # Check if argument labels is empty
        try:
            if not bool(labels):
                raise ValueError
        except ValueError:
            return(print("Value error: Variable labels is empty")) 


        sens_conf = {}
        for i in labels:
            sens_conf[labels[i]] = confusion_matrix(list(sens_df[labels[i]]['t']), list(sens_df[labels[i]]['p']), labels=[0,1]).ravel()

        true_positive_rate = {}
        false_positive_rate = {}
        true_negative_rate = {}
        false_negative_rate = {}

        for i in labels:
            true_positive_rate[labels[i]] = (sens_conf[labels[i]][3]/(sens_conf[labels[i]][3]+sens_conf[labels[i]][2]))
            false_positive_rate[labels[i]] = (sens_conf[labels[i]][1]/(sens_conf[labels[i]][1]+sens_conf[labels[i]][0]))
            true_negative_rate[labels[i]] = 1 - false_positive_rate[labels[i]]
            false_negative_rate[labels[i]] = 1 - true_positive_rate[labels[i]]

        return(true_positive_rate,false_positive_rate,true_negative_rate,false_negative_rate)



    def ability_plots(self,labels,TPR,FPR,TNR,FNR):
        """
        
        Creates plots representing the true positive, false positive, true
        negative and false negative scores.

        Parameters
        ----------
        labels : dict
            Dictionary containing the labels of the sensitive variables.
        TPR : dict
            dictionnary containing true postive rates.
        FPR : dict
            dictionnary containing false positive rates.
        TNR : dict
            dictionnary containing true negative rates.
        FNR : dict
            dictionnary containing false negative rates.

        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        ValueError
            When the dict of labels is empty.

        Returns
        -------
        None.

        """
        try:
            if not isinstance(labels, dict):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable labels should be of type dict"))
        
        try:
            if not (isinstance(TPR, dict) and isinstance(FPR, dict) and isinstance(TNR, dict) and isinstance(FNR, dict)):
                raise TypeError
        except TypeError:
            return(print("Type error: The ratios must be of type dict"))
           
        # Check if argument labels is empty
        try:
            if not bool(labels):
                raise ValueError
        except ValueError:
            return(print("Value error: Variable labels is empty")) 

        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=("True Positive Rate", "False Positive Rate", "True Negative Rate", "False Negative Rate"))

        x_axis = list(labels.values())
        fig.add_trace(
            go.Bar(x = x_axis, y=list(TPR.values())),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(FPR.values())),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(TNR.values())),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(FNR.values())),
            row=2, col=2
        )

        fig.update_layout(showlegend=False,height=600, width=800, title_text="Ability Disparities")
        fig.show()

    def ability_metrics(self,TPR,FPR,TNR,FNR):
        """
        
        Internal method that performs chi-squared test of the rates and prints if the 
        null hypothesis is rejected depending on the p-values.

        Parameters
        ----------
        TPR : dict
            dictionnary containing true postive rates.
        FPR : dict
            dictionnary containing false positive rates.
        TNR : dict
            dictionnary containing true negative rates.
        FNR : dict
            dictionnary containing false negative rates.

        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.

        Returns
        -------
        None.

        """
        
        try:
            if not (isinstance(TPR, dict) and isinstance(FPR, dict) and isinstance(TNR, dict) and isinstance(FNR, dict)):
                raise TypeError
        except TypeError:
            return(print("Type error: The ratios must be of type dict"))



        TPR_p = chisquare(list(np.array(list(TPR.values()))*100))[1]
        FPR_p = chisquare(list(np.array(list(FPR.values()))*100))[1]
        TNR_p = chisquare(list(np.array(list(TNR.values()))*100))[1]
        FNR_p = chisquare(list(np.array(list(FNR.values()))*100))[1]

        if TPR_p <= 0.01:
            print("*** Reject H0: Significant True Positive Disparity with p=",TPR_p)
        elif TPR_p <= 0.05:
            print("** Reject H0: Significant True Positive Disparity with p=",TPR_p)
        elif TPR_p <= 0.1:
            print("*  Reject H0: Significant True Positive Disparity with p=",TPR_p)
        else:
            print("Accept H0: True Positive Disparity Not Detected. p=",TPR_p)

        if FPR_p <= 0.01:
            print("*** Reject H0: Significant False Positive Disparity with p=",FPR_p)
        elif FPR_p <= 0.05:
            print("** Reject H0: Significant False Positive Disparity with p=",FPR_p)
        elif FPR_p <= 0.1:
            print("*  Reject H0: Significant False Positive Disparity with p=",FPR_p)
        else:
            print("Accept H0: False Positive Disparity Not Detected. p=",FPR_p)

        if TNR_p <= 0.01:
            print("*** Reject H0: Significant True Negative Disparity with p=",TNR_p)
        elif TNR_p <= 0.05:
            print("** Reject H0: Significant True Negative Disparity with p=",TNR_p)
        elif TNR_p <= 0.1:
            print("*  Reject H0: Significant True Negative Disparity with p=",TNR_p)
        else:
            print("Accept H0: True Negative Disparity Not Detected. p=",TNR_p)

        if FNR_p <= 0.01:
            print("*** Reject H0: Significant False Negative Disparity with p=",FNR_p)
        elif FNR_p <= 0.05:
            print("** Reject H0: Significant False Negative Disparity with p=",FNR_p)
        elif FNR_p <= 0.1:
            print("*  Reject H0: Significant False Negative Disparity with p=",FNR_p)
        else:
            print("Accept H0: False Negative Disparity Not Detected. p=",FNR_p)


        # Storing results in staging intermediate area results_stored dataframe
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'identify_bias', 
                                       str(self.ml_model_desc),'Chi2 test on TPR - Ability',str(TPR_p), ' ']
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'identify_bias', 
                                       str(self.ml_model_desc),'Chi2 test on FPR - Ability',str(FPR_p), ' ']
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'identify_bias', 
                                       str(self.ml_model_desc),'Chi2 test on TNR - Ability',str(TNR_p), ' ']
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'identify_bias', 
                                       str(self.ml_model_desc),'Chi2 test on FNR - Ability',str(FNR_p), ' ']
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)
            


    def predictive(self,labels,sens_df):
        """
        
        Internal method that calculates precision scores for each one of the labels in 
        the sensitive group, plots them in a bar plot and calculates the statistical 
        chi squared test.
        
        Parameters
        ----------
        labels : dict
            Dictionary containing the labels of the sensitive variables.
        sens_df : dict
            Dictionary containing a DataFrame for each subgroup of sensitive variables.
            
        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        ValueError
            When the dict labels or sens_df are empty.

        Returns
        -------
        Tuple with three elements :
            precision_dic : dict
                Dictionary of precision scores.
            fig : plotly.graph_objects.Figure
                Bar plot representing the precision scores.
            pred_p : 
                Chi squared test of the precision scores.
            

        """
        
        try:
            if not isinstance(sens_df, dict):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable sens_df should be of type dict"))
      
        try:
            if not isinstance(labels, dict):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable labels should be of type dict"))
        
        # Check if argument labels is empty
        try:
            if not bool(labels):
                raise ValueError
        except ValueError:
            return(print("Value error: Variable labels is empty")) 
        
        # Check if argument sens_df is empty
        try:
            if not bool(sens_df):
                raise ValueError
        except ValueError:
            return(print("Value error: Variable sens_df is empty")) 


        precision_dic = {}

        for i in labels:
            precision_dic[labels[i]] = precision_score(sens_df[labels[i]]['t'],sens_df[labels[i]]['p'])

        fig = go.Figure([go.Bar(x=list(labels.values()), y=list(precision_dic.values()))])

        pred_p = chisquare(list(np.array(list(precision_dic.values()))*100))[1]

        return(precision_dic,fig,pred_p)




    def identify_bias(self, sensitive,labels):
        """
        
        Performs the bias analysis of the datatset with respect to the sensitive 
        variable and its subgroups or labels and prints the results with the 
        following structure :
            REPRESENTATION
                Provides information regarding how the subgroups and response 
                labels are represented in the dataset and if the null hypothesis 
                is rejected for each of the chi squared tests.
                Uses the internal method representation.
            ABILITY
                Calculates the true positive rate, false positive rate, true 
                negative rate and false negative rate, performs the chi-squared 
                test of those rates and prints if the null hypothesis is rejected 
                depending on the p-values.
                Uses the internal methods ability, ability_plots and ability_metrics.
            PREDICTIVE
                calculates precision scores for each one of the labels in the 
                sensitive group, plots them in a bar plot, calculates the statistical 
                chi squared test and concludes whether the null hypothesis can be 
                rejected or not.
                Uses the internal method predictive.
        The method also checks if the requested calculations regarding identify_bias
        are already stored in the persistent sqlite3 database. If so, a prompt
        asking user if still wants to continue appears. If user selects "N", 
        the method returns a print statement and stops. If user selects any 
        other key, method proceeds with identify_bias calculations.
        
        Parameters
        ----------
        sensitive : str
            Name of the sensitive variable from which the labels will be 
            generated.
        labels : dict
            Dictionary containing the labels of the sensitive variables.
        
        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        ValueError
            When the dict labels is empty.

        Returns
        -------
        None.

        """
        try:
            if not isinstance(sensitive, str):
                 raise TypeError
        except TypeError:
            return(print("Type error: Variable sensitive should be of type str")) 
        
        try:
            if not isinstance(labels, dict):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable labels should be of type dict"))
            
        # Check if argument labels is empty
        try:
            if not bool(labels):
                raise ValueError
        except ValueError:
            return(print("Value error: Variable labels is empty"))   


        
        # Create model details to compare with database stored values
        if self.ml_model_object is not None:
            self.ml_model_desc2 = self.ml_model_object
        elif (self.ml_model_object is None) & (self.ml_prebuilt_model is not None):
            if self.ml_prebuilt_model == "LOG":
                self.ml_model_desc2 = LogisticRegression()
            if self.ml_prebuilt_model == "XGB":  
                self.ml_model_desc2 = xgboost.XGBClassifier()
            if self.ml_prebuilt_model == "MLP":  
                self.ml_model_desc2 = MLPClassifier()        
            if self.ml_prebuilt_model == "RFC":  
                self.ml_model_desc2 = RandomForestClassifier()
            if self.ml_prebuilt_model == "DTC":  
                self.ml_model_desc2 = DecisionTreeClassifier()


        # Check if data has been already calculated earlier
        engine = sqlalchemy.create_engine('sqlite:///FairDetectDataBase.db')
        queryF = "SELECT * FROM fairdetect_results \
                    WHERE filename = "+ "'"+self.data_filename+"'"+\
                    " AND target_var = "+"'"+self.target_var+"'"+\
                    " AND method = 'identify_bias'"+ \
                    " AND ml_model = "+'"'+str(self.ml_model_desc2)+'" '
        resultquery = pd.read_sql(queryF, engine)
        if resultquery.shape[0]>0:
            print("Found in persistent database equivalent ", resultquery.shape[0]," results")
            print("Criteria: same filename, same target_var, same method, same ml model")
            print("Showing first 20 results in SQLite database with this criteria")
            display(resultquery[0:20])
            value = input("Please press 'N' to stop method calculation, any other key to continue:")
            if value =='N': return print("Calculation of method stopped by user. Data can be retrieved from database using 'read_results_from_db' method ")
        
        predictions = self.model.predict(self.X_test)
        cont_table,sens_df,rep_fig,rep_p = self.representation(self.X_test,self.y_test,sensitive,labels,predictions)

        print("REPRESENTATION")
        rep_fig.show()

        print(cont_table,'\n')

        if rep_p <= 0.01:
            print("*** Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
            comment_str = str("*** Reject H0: Significant Relation Between "+sensitive+" and Target with p = "+str(rep_p))

        elif rep_p <= 0.05:
            print("** Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
            comment_str = str("** Reject H0: Significant Relation Between "+sensitive+" and Target with p = "+str(rep_p))
            
        elif rep_p <= 0.1:
            print("* Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
            comment_str = str("* Reject H0: Significant Relation Between "+sensitive+" and Target with p = "+str(rep_p))
            
        else:
            print("Accept H0: No Significant Relation Between",sensitive,"and Target Detected. p=",rep_p)
            comment_str = str("** Accept H0: No Significant Relation Between "+sensitive+" and Target Detected with p = "+str(rep_p))

        # Storing results in staging intermediate area results_stored dataframe
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'identify_bias', 
                                       str(self.ml_model_desc),'p-value chi2 cont table - Representation',rep_p, comment_str]
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)
            
        TPR, FPR, TNR, FNR = self.ability(sens_df,labels)
        print("\n\nABILITY")
        self.ability_plots(labels,TPR,FPR,TNR,FNR)
        self.ability_metrics(TPR,FPR,TNR,FNR)
        
        # Storing results in staging intermediate area results_stored dataframe
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'identify_bias', 
                                       str(self.ml_model_desc),'TPR - Ability',str(TPR), 'Confusion Matrix Metrics - Ability']
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'identify_bias', 
                                       str(self.ml_model_desc),'FPR - Ability',str(FPR), 'Confusion Matrix Metrics - Ability']
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'identify_bias', 
                                       str(self.ml_model_desc),'TNR - Ability',str(TNR), 'Confusion Matrix Metrics - Ability']
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'identify_bias', 
                                       str(self.ml_model_desc),'FNR - Ability',str(FNR), 'Confusion Matrix Metrics - Ability']
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)
        


        precision_dic, pred_fig, pred_p = self.predictive(labels,sens_df)
        print("\n\nPREDICTIVE")
        pred_fig.show()

        if pred_p <= 0.01:
            print("*** Reject H0: Significant Predictive Disparity with p=",pred_p)
            comment_str2 = str("*** Reject H0: Significant Predictive Disparity with p= "+str(pred_p))
        elif pred_p <= 0.05:
            print("** Reject H0: Significant Predictive Disparity with p=",pred_p)
            comment_str2 = str("** Reject H0: Significant Predictive Disparity with p= "+str(pred_p))
        elif pred_p <= 0.1:
            print("* Reject H0: Significant Predictive Disparity with p=",pred_p)
            comment_str2 = str("* Reject H0: Significant Predictive Disparity with p= "+str(pred_p))
        else:
            print("Accept H0: No Significant Predictive Disparity. p=",pred_p)
            comment_str2 = str("Accept H0: No Significant Predictive Disparity. p= "+str(pred_p))

        # Storing results in staging intermediate area results_stored dataframe
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'identify_bias', 
                                       str(self.ml_model_desc),'p-value chi2 precision - Predictive',pred_p, comment_str2]
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)


    
    def understand_shap(self,labels,sensitive,affected_group,affected_target):
        """
        
    	This method calculates and analyzes the SHAP values, which represent 
    	each feature importance or responsibility in the model output
    	Provides 5 different figures plotting the results derived from 
    	SHAP values.
             
        
        Parameters
        ----------

        labels : dict
            Dictionary containing the labels of the sensitive variables.
        sensitive : str
            Name of the sensitive variable from which the labels will be 
            generated.     
        affected_group : same type as labels.keys (str or int)
            Subgroup that is suspected to be subject to bias.
        affected_target : int
            Value of the affected label for the response variable.

        Raises
        ------
        TypeError
            When the method is called with arguments of the wrong type.
        ValueError
            When the list of labels is empty.
        Exception
            When the variable affected_group is not in the sensitive column values
        TypeError
            When an unsupported model by shap.Explainer is used

        Returns
        -------
        None.

        """
        
        try:
            if not isinstance(labels, dict):
                raise TypeError
        except TypeError:
            return(print("Type error: Variable labels should be of type dict"))
                        
        try:
            if not isinstance(sensitive, str):
                 raise TypeError
        except TypeError:
            return(print("Type error: Variable sensitive should be of type str"))
            
        try:
            if not isinstance(affected_group, (str, int)):
                 raise TypeError
        except TypeError:
            return(print("Type error: Variable affected_group should be of type str or int \
                         (should be of the same type as labels.keys)"))
            
        try:
            if not isinstance(affected_target, int):
                 raise TypeError
        except TypeError:
            return(print("Type error: Variable affected_target should be of type int"))
            
        # Check if argument labels is empty
        try:
            if not bool(labels):
                raise ValueError
        except ValueError:
            return(print("Value error: Variable labels is empty"))
        
        # Check if affected_group is in set(self.X_test[sensitive])
        try:
            if not affected_group in set(self.X_test[sensitive]):
                raise Exception
        except Exception:
            return(print("Exception: Variable affected_group is not in the sensitive column values"))

        # Check if the model is supported by shap.Explainer
        try:
            shap.Explainer(self.model)
        except TypeError:
            return(print("The provided model is not supported by the shap explainer function"))

        
        
        # Create model details to compare with database stored values
        if self.ml_model_object is not None:
            self.ml_model_desc2 = self.ml_model_object
        elif (self.ml_model_object is None) & (self.ml_prebuilt_model is not None):
            if self.ml_prebuilt_model == "LOG":
                self.ml_model_desc2 = LogisticRegression()
            if self.ml_prebuilt_model == "XGB":  
                self.ml_model_desc2 = xgboost.XGBClassifier()
            if self.ml_prebuilt_model == "MLP":  
                self.ml_model_desc2 = MLPClassifier()        
            if self.ml_prebuilt_model == "RFC":  
                self.ml_model_desc2 = RandomForestClassifier()
            if self.ml_prebuilt_model == "DTC":  
                self.ml_model_desc2 = DecisionTreeClassifier()


        # Check if data has been already calculated earlier
        engine = sqlalchemy.create_engine('sqlite:///FairDetectDataBase.db')
        queryF = "SELECT * FROM fairdetect_results \
                    WHERE filename = "+ "'"+self.data_filename+"'"+\
                    " AND target_var = "+"'"+self.target_var+"'"+\
                    " AND method = 'understand_shap'"+ \
                    " AND ml_model = "+'"'+str(self.ml_model_desc2)+'" '
        resultquery = pd.read_sql(queryF, engine)
        if resultquery.shape[0]>0:
            print("Found in persistent database equivalent ", resultquery.shape[0]," results")
            print("Criteria: same filename, same target_var, same method, same ml model")
            print("Showing first 20 results in SQLite database with this criteria")
            display(resultquery[0:20])
            value = input("Please press 'N' to stop method calculation, any other key to continue:")
            if value =='N': return print("Calculation of method stopped by user. Data can be retrieved from database using 'read_results_from_db' method ")
        
        
        explainer = shap.Explainer(self.model)

        full_table = self.X_test.copy()
        full_table['t'] = self.y_test
        full_table['p'] = self.model.predict(self.X_test)
        full_table

        shap_values = explainer(self.X_test)

        sens_glob_coh = np.where(self.X_test[sensitive]==list(labels.keys())[0],labels[0],labels[1])

        misclass = full_table[full_table.t != full_table.p]
        affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]
        shap_values2 = explainer(affected_class.drop(['t','p'],axis=1))
        
        figure,axes = plt.subplots(nrows=2, ncols=2,figsize=(20,10))
        plt.subplots_adjust(right=1.4,wspace=1)

        print("Model Importance Comparison")
        plt.subplot(1, 2, 1) # row 1, col 2 index 1   
        shap.plots.bar(shap_values.cohorts(sens_glob_coh).abs.mean(0),show=False)
        plt.legend(loc='lower right')#, bbox_to_anchor=(0.5, -0.05))#,plt.legend(loc='lower right')
        plt.subplot(1, 2, 2) # row 1, col 2 index 1
        shap_values2 = explainer(affected_class.drop(['t','p'],axis=1))
        shap.plots.bar(shap_values2)
        
        full_table['t'] = self.y_test
        full_table['p'] = self.model.predict(self.X_test)

        misclass = full_table[full_table.t != full_table.p]
        affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]

        truclass = full_table[full_table.t == full_table.p]
        tru_class = truclass[(truclass[sensitive] == affected_group) & (truclass.t == affected_target)]

        x_axis = list(affected_class.drop(['t','p',sensitive],axis=1).columns)
        affect_character = list((affected_class.drop(['t','p',sensitive],axis=1).mean()-tru_class.drop(['t','p',sensitive],axis=1).mean())/affected_class.drop(['t','p',sensitive],axis=1).mean())

        dict1tostore = dict(zip(x_axis, affect_character))
        
        # Storing results in staging intermediate area results_stored dataframe
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'understand_shap', 
                                       str(self.ml_model_desc),'Average Affected Attribute Comparison to True Class Members',
                                       str(dict1tostore),'Dictionary with column names as key and result as values']
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)

        
        fig = go.Figure([go.Bar(x=x_axis, y=affect_character)])

        print("Affected Attribute Comparison")
        print("Average Comparison to True Class Members")
        fig.show()

        misclass = full_table[full_table.t != full_table.p]
        affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]

        tru_class = full_table[(full_table[sensitive] == affected_group) & (full_table.p == affected_target)]

        x_axis = list(affected_class.drop(['t','p',sensitive],axis=1).columns)
        affect_character = list((affected_class.drop(['t','p',sensitive],axis=1).mean()-full_table.drop(['t','p',sensitive],axis=1).mean())/affected_class.drop(['t','p',sensitive],axis=1).mean())

        dict2tostore = dict(zip(x_axis, affect_character))

        # Storing results in staging intermediate area results_stored dataframe
        self.results_stored.loc[-1] = [self.data_filename,self.target_var, datetime.now(), 'understand_shap', 
                                       str(self.ml_model_desc),'Average Affected Attribute Comparison to All Members',
                                       str(dict2tostore),'Dictionary with column names as key and result as values']
        self.results_stored.index = self.results_stored.index + 1  # shifting index
        self.results_stored.sort_index(inplace=True)

        
        
        fig = go.Figure([go.Bar(x=x_axis, y=affect_character)])
        print("Average Comparison to All Members")
        fig.show()

        print("Random Affected Decision Process")
        
        explainer = shap.Explainer(self.model)
        
        shap.plots.waterfall(explainer(affected_class.drop(['t','p'],axis=1))[randrange(0, len(affected_class))],show=False)


