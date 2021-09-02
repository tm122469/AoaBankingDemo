from teradataml import create_context, DataFrame, copy_to_sql, remove_context
from aoa.stats import stats
from aoa.util.artefacts import save_plot
from teradataml.analytics.valib import *
from teradataml import configure

import os

configure.val_install_location = os.environ.get("AOA_VAL_DB", "VAL")

def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    features_table = data_conf["features"]
    
    ads = DataFrame(features_table)

    print("Starting training...")
    
    feature_names = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents',
                     'TenureMonths',
                     'PhoneService',
                     'PaperlessBilling',
                     'MonthlyCharges',
                     'TotalCharges',
                     'CLTV',
                     'Age',
                     'Under30',
                     'Married',
                     'NumberOfDependents',
                     'ReferredAFriend',
                     'NumberOfReferrals',
                     'AvgMonthlyLongDistanceCharges',
                     'AvgMonthlyGBDownload',
                     'DeviceProtectionPlan',
                     'PremiumTechSupport',
                     'StreamingMusic',
                     'UnlimitedData',
                     'TotalRefunds',
                     'TotalExtraDataCharges',
                     'TotalLongDistanceCharges',
                     'TotalRevenue',
                     'SatisfactionScore',
                     'MultipleLines_No',
                     'MultipleLines_Yes',
                     'InternetService_DSL',
                     'InternetService_FiberOptic',
                     'OnlineSecurity_No',
                     'OnlineSecurity_Yes',
                     'OnlineBackup_No',
                     'OnlineBackup_Yes',
                     'DeviceProtection_No',
                     'DeviceProtection_Yes',
                     'TechSupport_No',
                     'TechSupport_Yes',
                     'StreamingTV_No',
                     'StreamingTV_Yes',
                     'StreamingMovies_No',
                     'StreamingMovies_Yes',
                     'Contract_OneYear',
                     'Contract_TwoYear',
                     'PaymentMethod_AutoBankTransfer',
                     'PaymentMethod_AutoCreditCard',
                     'PaymentMethod_ECheck',
                     'InternetType_Cable',
                     'InternetType_DSL',
                     'InternetType_FiberOptic',
                     'Offer_OfferA',
                     'Offer_OfferB',
                     'Offer_OfferC',
                     'Offer_OfferD',
                     'Offer_OfferE']

    target_name = "ChurnValue"

    model = valib.LogReg(data=ads, 
                         columns=feature_names, 
                         response_column=target_name, 
                         response_value=1,
                         threshold_output='true',
                         near_dep_report='true', 
                         cond_ind_threshold=int(hyperparams["cond_ind_threshold"]),
                         variance_prop_threshold=float(hyperparams["variance_prop_threshold"]))
   
    model.model.to_sql(table_name=kwargs.get("model_table"), if_exists="replace")
    model.statistical_measures.to_sql(table_name = kwargs.get("model_table") + "_rpt", if_exists = 'replace')
    
    print("Finished training")

    print("Calculating dataset statistics")
    
    stats.record_training_stats(ads,
                       features=feature_names,
                       predictors=[target_name],
                        # bug in VAL frequency won't allow us to specify more categorical columns
                        # tracked in https://github.com/ThinkBigAnalytics/AoaPythonClient/issues/155 - Ask Anton about using "All Categorical" option?
                       categorical=[target_name, feature_names],
                       category_labels={target_name: {0: "false", 1: "true"}})
    
    print("Finished calculating dataset statistics")
    
    remove_context()
    
