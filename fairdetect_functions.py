import matplotlib.pyplot as plt
from random import randrange
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dalex as dx
import pandas as pd
from scipy.stats import chi2_contingency
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from scipy.stats import chisquare
from sklearn.metrics import precision_score

def create_labels(X_test,sensitive):
    sensitive_label = {}
    for i in set(X_test[sensitive]):
        text = "Please Enter Label for Group" +" "+ str(i)+": "
        label = input(text)
        sensitive_label[i]=label
    return(sensitive_label)


def representation(X_test,y_test,sensitive,labels,predictions):
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
        


def ability(sens_df,labels):
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



def ability_plots(labels,TPR,FPR,TNR,FNR):
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

def ability_metrics(TPR,FPR,TNR,FNR):
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




def predictive(labels,sens_df):
    precision_dic = {}

    for i in labels:
        precision_dic[labels[i]] = precision_score(sens_df[labels[i]]['t'],sens_df[labels[i]]['p'])

    fig = go.Figure([go.Bar(x=list(labels.values()), y=list(precision_dic.values()))])
    
    pred_p = chisquare(list(np.array(list(precision_dic.values()))*100))[1]
    
    return(precision_dic,fig,pred_p)




def identify_bias(model,X_test,y_test,sensitive,labels):
    predictions = model.predict(X_test)
    cont_table,sens_df,rep_fig,rep_p = representation(X_test,y_test,sensitive,labels,predictions)

    print("REPRESENTATION")
    rep_fig.show()

    print(cont_table,'\n')

    if rep_p <= 0.01:
        print("*** Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
    elif rep_p <= 0.05:
        print("** Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
    elif rep_p <= 0.1:
        print("* Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
    else:
        print("Accept H0: No Significant Relation Between",sensitive,"and Target Detected. p=",rep_p)

    TPR, FPR, TNR, FNR = ability(sens_df,labels)
    print("\n\nABILITY")
    ability_plots(labels,TPR,FPR,TNR,FNR)
    ability_metrics(TPR,FPR,TNR,FNR)


    precision_dic, pred_fig, pred_p = predictive(labels,sens_df)
    print("\n\nPREDICTIVE")
    pred_fig.show()

    if pred_p <= 0.01:
        print("*** Reject H0: Significant Predictive Disparity with p=",pred_p)
    elif pred_p <= 0.05:
        print("** Reject H0: Significant Predictive Disparity with p=",pred_p)
    elif pred_p <= 0.1:
        print("* Reject H0: Significant Predictive Disparity with p=",pred_p)
    else:
        print("Accept H0: No Significant Predictive Disparity. p=",pred_p)


def understand_shap(X_test,y_test,model,labels,sensitive,affected_group,affected_target):
    import shap
    explainer = shap.Explainer(model)
    
    full_table = X_test.copy()
    full_table['t'] = y_test
    full_table['p'] = model.predict(X_test)
    full_table

    shap_values = explainer(X_test)
    sens_glob_coh = np.where(X_test[sensitive]==list(labels.keys())[0],labels[0],labels[1])
    
    misclass = full_table[full_table.t != full_table.p]
    affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]
    shap_values2 = explainer(affected_class.drop(['t','p'],axis=1))
    #sens_mis_coh = np.where(affected_class[sensitive]==list(labels.keys())[0],labels[0],labels[1])


    figure,axes = plt.subplots(nrows=2, ncols=2,figsize=(20,10))
    plt.subplots_adjust(right=1.4,wspace=1)
    
    print("Model Importance Comparison")
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    shap.plots.bar(shap_values.cohorts(sens_glob_coh).abs.mean(0),show=False)
    plt.subplot(1, 2, 2) # row 1, col 2 index 1
    shap_values2 = explainer(affected_class.drop(['t','p'],axis=1))
    shap.plots.bar(shap_values2)
    #shap.plots.bar(shap_values2)
    
    full_table['t'] = y_test
    full_table['p'] = model.predict(X_test)
    #full_table=full_table[['checking_account','credit_amount','duration','sex','t','p']]

    misclass = full_table[full_table.t != full_table.p]
    affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]

    truclass = full_table[full_table.t == full_table.p]
    tru_class = truclass[(truclass[sensitive] == affected_group) & (truclass.t == affected_target)]

    x_axis = list(affected_class.drop(['t','p',sensitive],axis=1).columns)
    affect_character = list((affected_class.drop(['t','p',sensitive],axis=1).mean()-tru_class.drop(['t','p',sensitive],axis=1).mean())/affected_class.drop(['t','p',sensitive],axis=1).mean())

    #plt.figsize([10,10])
    #plt.bar(x_axis,affect_character)

    fig = go.Figure([go.Bar(x=x_axis, y=affect_character)])
    
    print("Affected Attribute Comparison")
    print("Average Comparison to True Class Members")
    fig.show()
    
    misclass = full_table[full_table.t != full_table.p]
    affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]

    #truclass = full_table[full_table.t == full_table.p]
    tru_class = full_table[(full_table[sensitive] == affected_group) & (full_table.p == affected_target)]

    x_axis = list(affected_class.drop(['t','p',sensitive],axis=1).columns)
    affect_character = list((affected_class.drop(['t','p',sensitive],axis=1).mean()-full_table.drop(['t','p',sensitive],axis=1).mean())/affected_class.drop(['t','p',sensitive],axis=1).mean())

    #plt.figsize([10,10])
    #plt.bar(x_axis,affect_character)

    fig = go.Figure([go.Bar(x=x_axis, y=affect_character)])
    print("Average Comparison to All Members")
    fig.show()
    
    print("Random Affected Decision Process")
    explainer = shap.Explainer(model)
    shap.plots.waterfall(explainer(affected_class.drop(['t','p'],axis=1))[randrange(0, len(affected_class))],show=False)

    
    