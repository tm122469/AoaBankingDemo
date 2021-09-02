from sklearn import metrics
from teradataml import create_context, DataFrame, copy_to_sql, remove_context, INTEGER
from aoa.stats import stats
from aoa.util.artefacts import save_plot
from teradataml.analytics.valib import *
from teradataml import configure

import os
import matplotlib.pyplot as plt
import itertools
import json

configure.val_install_location = os.environ.get("AOA_VAL_DB", "VAL")

def evaluate(data_conf, model_conf, **kwargs):
    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    features_table = data_conf["features"]
    
    ads = DataFrame(features_table)
    model = DataFrame(kwargs.get("model_table"))
    
    score = valib.LogRegPredict(data=ads, 
                                model=model, 
                                index_columns="CustomerID",
                                estimate_column="PredictChurnValue",
                                prob_column="Probability",
                                accumulate="ChurnValue")
    
    results = score.result
    results = results.assign(PredictChurnValue=results.PredictChurnValue.cast(type_=INTEGER))
    
    predictions = results.select(["ChurnValue", "PredictChurnValue"]).to_pandas()
    
    y_pred = predictions[["PredictChurnValue"]]
    y_test = predictions[["ChurnValue"]]
    
    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }
    
    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    # create confusion matrix plot
    cf = metrics.confusion_matrix(y_test, y_pred)

    plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['0','1'])
    plt.yticks([0, 1], ['0','1'])

    thresh = cf.max() / 2.
    for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
        plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')

    fig = plt.gcf()
    fig.savefig('artifacts/output/confusion_matrix', dpi=500)
    plt.clf()

    print("Evaluation complete...")

    print("Calculating dataset statistics")
    
    # the number of rows output from VAL is different to the number of input rows.. nulls???
    # temporary workaround - join back to features and filter features without predictions???
    
    results.to_sql(table_name="telco_predictions_tmp", if_exists='replace', temporary=True)
    ads = DataFrame.from_query(f"SELECT F.* FROM {features_table} F JOIN telco_predictions_tmp P ON F.CustomerID = P.CustomerID")
    
    stats.record_evaluation_stats(ads, results)
    
    print("Finished calculating dataset statistics")
    
    remove_context()
