import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.from sklearn.feature_extraction import DictVectorizer
import pandas as pd

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(df, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    print("Starting...")

    mlflow.set_tracking_uri(uri="http://mlflow:5000")
    mlflow.experiment_name("lr_model_mage")
    with mlflow.start_run():
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        train_dicts = df[categorical].to_dict(orient='records')
        print("Computing vectorization...")
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        print("Vectorization computed")
        target = 'duration'
        y_train = df[target].values
        print("Computing model...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        mlflow.sklearn.log_model(sk_model=lr,
                            "linear_regression_model",
                            input_example=X_train)
        mlflow.log_artifact(dv, "dict_vectorizer")

        y_pred = lr.predict(X_train)

    print("Model intercept is: ", lr.intercept_)
    return lr, dv
