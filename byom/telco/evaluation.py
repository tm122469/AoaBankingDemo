from sklearn import metrics
from teradataml import create_context
from teradataml.context.context import get_connection
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats

import os
import json
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy

def evaluate(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    model_id = kwargs["model_id"]

    engine = create_context(host=os.environ["AOA_CONN_HOST"],
                            username=os.environ["AOA_CONN_USERNAME"],
                            password=os.environ["AOA_CONN_PASSWORD"])

    cursor = engine.raw_connection().cursor()
    conn = get_connection()

    with open("artifacts/input/model.pmml", "rb") as f:
        model_bytes = f.read()

    
    cursor.execute("delete from AOA_Demo.pmml_models where model_id = 'telco_churn_byom'")
    modelname_param = "telco_churn_byom"
    cursor.execute(f"insert into AOA_Demo.pmml_models (model_id, model) VALUES (?,?)",
                  (modelname_param, model_bytes))


    scores_df = pd.read_sql(f"""
                SELECT CustomerID, ChurnValue as y_test, CAST(CAST(score_result AS JSON).JSONExtractValue('$.predicted_ChurnValue') as INT) as y_pred FROM IVSM.IVSM_SCORE(
                  ON (SELECT * FROM {data_conf["features"]}) AS DataTable
                  ON (SELECT * FROM AOA_Demo.pmml_models WHERE model_id = 'telco_churn_byom') AS ModelTable DIMENSION
                  USING 
                    ModelID('telco_churn_byom')
                    ColumnsToPreserve('CustomerID', 'ChurnValue')
                    ModelType('PMML')
                    ) sc;
                """, conn)
    
    y_pred = scores_df[["y_pred"]]
    y_test = scores_df[["y_test"]]

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

    predictions_table = "telco_churn_scores"
    predictions_df = scores_df[["CustomerID", "y_pred"]].rename({'y_pred': 'ChurnValue'}, axis=1)
    copy_to_sql(df=predictions_df, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    # the number of rows output from VAL is different to the number of input rows.. nulls?
    # temporary workaround - join back to features and filter features without predictions
    ads = DataFrame.from_query(f"SELECT F.* FROM {data_conf['features']} F JOIN telco_churn_scores P ON F.CustomerID = P.CustomerID")

    stats.record_evaluation_stats(ads, DataFrame(predictions_table))
