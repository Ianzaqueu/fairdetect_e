
Python package to analyze bias in dataset and model created by IE MBD22 Group E. Based on work from Ryan Daher & Prof Manoel Gadi.

The class docstring is shown below:

class FairDetectE(pandas.core.frame.DataFrame)
 |  FairDetectE(pd_DF, data_filename=None, target_var=None, ml_model_object=None, ml_prebuilt_model=None, *args, **kwargs)
 |  
 |  Method resolution order:
 |      FairDetectE
 |      pandas.core.frame.DataFrame
 |      pandas.core.generic.NDFrame
 |      pandas.core.base.PandasObject
 |      pandas.core.accessor.DirNamesMixin
 |      pandas.core.indexing.IndexingMixin
 |      pandas.core.arraylike.OpsMixin
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  X_y_split(self, target_var=None)
 |      Split the given dataframe into two:
 |          y_data that has the "target" feature only,
 |          &
 |          X_data gat has all the features except the "target" feature.
 |          The two new dataframes have same number of rows.
 |      
 |      Parameters
 |      ----------
 |      target_var: str
 |          the "target" feature that will be used to split the given dataframe.
 |      
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      ValueError
 |          When the variable, the feature name, is empty.     
 |      
 |      Returns
 |      -------
 |      pandas dataframes
 |          The function returns two datasets, one containing the target variable
 |          and one containing all the remaining features.
 |  
 |  __init__(self, pd_DF, data_filename=None, target_var=None, ml_model_object=None, ml_prebuilt_model=None, *args, **kwargs)
 |      Method that initializes an object of the class fairdetect_group_e with the 
 |      given arguments.
 |      
 |      Parameters
 |      ----------
 |      pd_DF : pandas.DataFrame
 |          Input DataFrame with all the data that will be needed for the bias analysis.
 |      data_filename : str, optional
 |          User input name of pandas dataframe, to be stored in persistent
 |          sqlite3 database.The default is None.
 |      target_var : str, optional
 |          Name of the target variable. The default is None.
 |      ml_model_object : sklearn or xgboost, optional
 |          model object from sklearn or xgboost. The default is None.
 |      ml_prebuilt_model : str, optional
 |          Acronym of the ML algorithm that will be used to generate the prediction.
 |          Options: 'DTC', 'XGB', 'MLP', 'RFC', 'LOG'.The default is None.
 |      
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      Exception
 |          When the target_var is not present in the DataFrame.
 |          Or, when ml_prebuilt_model is not one of the supported ones:
 |              'DTC', 'XGB', 'MLP', 'RFC', 'LOG'
 |      
 |      Returns
 |      -------
 |      None.
 |  
 |  ability(self, sens_df, labels)
 |      Calculates the true positive rate, false positive rate, true negative rate
 |      and false negative rate for a given dictionary of labels and dataframe.
 |      
 |      Parameters
 |      ----------
 |      sens_df : dict
 |          Dictionary containing a DataFrame for each subgroup of sensitive variables.
 |      labels : dict
 |          Dictionary containing the labels of the sensitive variables.
 |          
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      ValueError
 |          When the dict of labels is empty.
 |      
 |      Returns
 |      -------
 |      true_positive_rate : dict
 |          dictionary containing the true positive rates
 |      false_positive_rate : dict
 |          dictionary containing the false positive rates
 |      true_negative_rate : dict
 |          dictionary containing the true negative rates
 |      false_negative_rate : dict
 |          dictionary containing the false negative rates
 |  
 |  ability_metrics(self, TPR, FPR, TNR, FNR)
 |      Internal method that performs chi-squared test of the rates and prints if the 
 |      null hypothesis is rejected depending on the p-values.
 |      
 |      Parameters
 |      ----------
 |      TPR : dict
 |          dictionnary containing true postive rates.
 |      FPR : dict
 |          dictionnary containing false positive rates.
 |      TNR : dict
 |          dictionnary containing true negative rates.
 |      FNR : dict
 |          dictionnary containing false negative rates.
 |      
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      
 |      Returns
 |      -------
 |      None.
 |  
 |  ability_plots(self, labels, TPR, FPR, TNR, FNR)
 |      Creates plots representing the true positive, false positive, true
 |      negative and false negative scores.
 |      
 |      Parameters
 |      ----------
 |      labels : dict
 |          Dictionary containing the labels of the sensitive variables.
 |      TPR : dict
 |          dictionnary containing true postive rates.
 |      FPR : dict
 |          dictionnary containing false positive rates.
 |      TNR : dict
 |          dictionnary containing true negative rates.
 |      FNR : dict
 |          dictionnary containing false negative rates.
 |      
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      ValueError
 |          When the dict of labels is empty.
 |      
 |      Returns
 |      -------
 |      None.
 |  
 |  create_labels(self, X_test, sensitive)
 |      Creates labels for the variable of which the user wants to detect bias
 |      
 |      Parameters
 |      ----------
 |      X_test : pandas.DataFrame
 |          DataFrame with the test data to be analysed.
 |      sensitive : str
 |          Name of the sensitive variable from which the labels will be 
 |          generated.
 |          
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      Exception
 |          When the sensitive label is not present in the DataFrame.
 |      
 |      Returns
 |      -------        
 |      sensitive_label : dict
 |          Dictionary with the sensitive labels
 |  
 |  encode_labels(self, df=None)
 |      Obtains features / attributes of object type from a dataset and encodes them into integers. When call it,
 |      a dataframe must be passed to it. 
 |      
 |      Parameters
 |      ----------
 |      df : pandas dataframe
 |          A pandas dataframe which has attributes of "object" type features to be encoded
 |          
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      ValueError
 |          When the dataframe is empty.     
 |         
 |      Returns
 |      -------
 |      TYPE pandas dataframe
 |          The function returns a pandas dataframe with all features of type integers or float.
 |  
 |  identify_bias(self, sensitive, labels)
 |      Performs the bias analysis of the datatset with respect to the sensitive 
 |      variable and its subgroups or labels and prints the results with the 
 |      following structure :
 |          REPRESENTATION
 |              Provides information regarding how the subgroups and response 
 |              labels are represented in the dataset and if the null hypothesis 
 |              is rejected for each of the chi squared tests.
 |              Uses the internal method representation.
 |          ABILITY
 |              Calculates the true positive rate, false positive rate, true 
 |              negative rate and false negative rate, performs the chi-squared 
 |              test of those rates and prints if the null hypothesis is rejected 
 |              depending on the p-values.
 |              Uses the internal methods ability, ability_plots and ability_metrics.
 |          PREDICTIVE
 |              calculates precision scores for each one of the labels in the 
 |              sensitive group, plots them in a bar plot, calculates the statistical 
 |              chi squared test and concludes whether the null hypothesis can be 
 |              rejected or not.
 |              Uses the internal method predictive.
 |      The method also checks if the requested calculations regarding identify_bias
 |      are already stored in the persistent sqlite3 database. If so, a prompt
 |      asking user if still wants to continue appears. If user selects "N", 
 |      the method returns a print statement and stops. If user selects any 
 |      other key, method proceeds with identify_bias calculations.
 |      
 |      Parameters
 |      ----------
 |      sensitive : str
 |          Name of the sensitive variable from which the labels will be 
 |          generated.
 |      labels : dict
 |          Dictionary containing the labels of the sensitive variables.
 |      
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      ValueError
 |          When the dict labels is empty.
 |      
 |      Returns
 |      -------
 |      None.
 |  
 |  predictive(self, labels, sens_df)
 |      Internal method that calculates precision scores for each one of the labels in 
 |      the sensitive group, plots them in a bar plot and calculates the statistical 
 |      chi squared test.
 |      
 |      Parameters
 |      ----------
 |      labels : dict
 |          Dictionary containing the labels of the sensitive variables.
 |      sens_df : dict
 |          Dictionary containing a DataFrame for each subgroup of sensitive variables.
 |          
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      ValueError
 |          When the dict labels or sens_df are empty.
 |      
 |      Returns
 |      -------
 |      Tuple with three elements :
 |          precision_dic : dict
 |              Dictionary of precision scores.
 |          fig : plotly.graph_objects.Figure
 |              Bar plot representing the precision scores.
 |          pred_p : 
 |              Chi squared test of the precision scores.
 |  
 |  read_results_from_db(self, sqlquery=None)
 |      Reads the results from the database. If no sqlquery is passed, a default query showing the 
 |      latest 10 results is shown. If a sqlquery is passed, the result from the query is returned as 
 |      a pandas dataframe.
 |      
 |      Parameters
 |      ----------
 |      sqlquery : str
 |          A string containing a correct SQL query
 |          
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |         
 |      Returns
 |      -------
 |      TYPE pandas dataframe
 |          The function returns a pandas dataframe with all features of type integers or float.
 |  
 |  representation(self, X_test, y_test, sensitive, labels, predictions)
 |      Internal method that returns cont_t, sens_df, fig and p, necessary for the 
 |      identify_bias method. It provides information regarding how the subgroups 
 |      and response labels are represented in the dataset.
 |      
 |      Parameters
 |      ----------
 |      X_test : pandas.DataFrame
 |          DataFrame containing the independent variables.
 |      y_test : pandas.DataFrame
 |          DataFrame containing the dependent variables.
 |      sensitive : str
 |          Sensitive variable.
 |      labels : dict
 |          dictionnary with all labels.
 |      predictions : np.array or pandas.DataFrame
 |          DataFrame containing the model predictions for the response variable.
 |      
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      Exception
 |          When the sensitive label is not present in the DataFrame.
 |      ValueError
 |          When the dict of labels is empty.
 |      
 |      Returns
 |      -------
 |      cont_table : tabulate(pandas.DataFrame)
 |          Table that shows the proportions of positive and negative
 |          outcomes for each of the subgroups in the sensitive variable.
 |      sens_df : dict
 |          Dictionary containing a DataFrame for each subgroup of sensitive variables
 |      fig : plotly.graph_objects
 |          Plotly subplots representing :
 |              - proportion of the different subgroups in the dataset
 |              - proportion of the different response labels in the dataset
 |      p : float
 |          the p-value of the chi square statistical test applied to the contingency 
 |          table.
 |  
 |  split_data_totrain(self, train_size_split=0.8, random_state_split=0)
 |      Split the given dataframe into two, based on a given split ratio:
 |      1 dataframe to train the model and 1 to test the model.
 |      
 |      Parameters
 |      ----------
 |      train_size_split: float
 |          The split ratio used to determine the size of each split for training and testing dataframes.
 |      random_state_split: int
 |          Ensures that the generated splits are reproducible.
 |      
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      ValueError
 |          When the variable, the feature name, is empty.     
 |      
 |      Returns
 |      -------
 |      pandas dataframes
 |          The function returns 4 dataframes, two for training and two for testing.
 |          First split contains training dataframes:
 |              X_train: contains all features except the target feature.
 |              y_train: contains target feature.
 |          Second split contains test dataframes:
 |              X_test: contains all features except the target feature.
 |              y_test: contains target feature.
 |  
 |  store_results_in_db(self)
 |      By calling this method, all the results obtained and automatically saved internally in the 
 |      temporary pd.DataFrame "results_stored" are saved into "FairDetectDataBase", table 
 |      "fairdetect_results". After saving into database in a persistent manner, the temporary 
 |      pd.DataFrame "results_stored" is reinitialized to an empty pandas DataFrame again.
 |      
 |      
 |      Parameters
 |      ----------
 |      No user input parameters.
 |          
 |      Raises
 |      ------
 |      None
 |      
 |      Returns
 |      -------
 |      None
 |  
 |  train_model(self, ml_prebuilt_model=None, ml_model_object=None)
 |      This method takes the user input machine learning model desired, and 
 |      trains it with the X_train dataset of the instance. 
 |      The user can provide prebuilt models ('DTC', 'XGB', 'MLP', 'RFC', 'LOG') 
 |      or pass a model object.
 |      
 |      Parameters
 |      ----------
 |      ml_prebuilt_model : str, optional
 |          Acronym of the ML algorithm that will be used to generate the prediction.
 |          Options: 'DTC', 'XGB', 'MLP', 'RFC', 'LOG'.The default is None.
 |      ml_model_object : sklearn or xgboost, optional
 |          model object from sklearn or xgboost. The default is None.
 |      
 |      
 |      Returns
 |      -------
 |      None.
 |  
 |  understand_shap(self, labels, sensitive, affected_group, affected_target)
 |      
 |	This method calculates and analyzes the SHAP values, which represent 
 |	each feature importance or responsibility in the model output
 |	Provides 5 different figures plotting the results derived from 
 |	SHAP values.
 |          
 |      
 |      Parameters
 |      ----------
 |      
 |      labels : dict
 |          Dictionary containing the labels of the sensitive variables.
 |      sensitive : str
 |          Name of the sensitive variable from which the labels will be 
 |          generated.     
 |      affected_group : same type as labels.keys (str or int)
 |          Subgroup that is suspected to be subject to bias.
 |      affected_target : int
 |          Value of the affected label for the response variable.
 |      
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      ValueError
 |          When the list of labels is empty.
 |      Exception
 |          When the variable affected_group is not in the sensitive column values
 |      TypeError
 |          When an unsupported model by shap.Explainer is used
 |      
 |      Returns
 |      -------
 |      None.
 |  
 |  ----------------------------------------------------------------------
 |  Class methods defined here:
 |  
 |  from_file(filename, target_var=None, ml_model_object=None, ml_prebuilt_model=None) from builtins.type
 |      Converts the input file into a DataFrame and calls the class __init__ method with it.
 |      
 |      Parameters
 |      ----------
 |      filename : str
 |          Name of the input file, must be one of the following types: csv, xls, xlsx, xml or json.
 |      target_var : str, optional
 |          Name of the target variable. The default is None.
 |      ml_model_object : sklearn or xgboost, optional
 |          model object from sklearn or xgboost. The default is None.
 |      ml_prebuilt_model : str, optional
 |          Acronym of the ML algorithm that will be used to generate the prediction.
 |          Options: 'DTC', 'XGB', 'MLP', 'RFC', 'LOG'.The default is None.
 |      
 |      Raises
 |      ------
 |      TypeError
 |          When the method is called with arguments of the wrong type.
 |      Exception
 |          When the target_var is not present in the DataFrame.
 |          Or, when ml_prebuilt_model is not one of the supported ones:
 |              'DTC', 'XGB', 'MLP', 'RFC', 'LOG'
 |      
 |      Returns
 |      -------
 |      Required information for __init__ to initialize instance of 
 |      class fairdetect_group_e
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  data_filename = None
 |  
 |  ml_model_object = None
 |  
 |  ml_prebuilt_model = None
 |  
 |  target_var = None