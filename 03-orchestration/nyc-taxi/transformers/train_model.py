@transformer
def execute(df, **kwargs):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import root_mean_squared_error
    import mlflow

    mlflow.set_tracking_uri("http://127.0.0.1:5500")
    mlflow.set_experiment("nyc-taxi-march-2023")

    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    y_train = df['duration'].values

    with mlflow.start_run() as run:
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        rmse = root_mean_squared_error(y_train, y_pred)
        mlflow.log_metric("rmse", rmse)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", categorical)

        # os.makedirs("models", exist_ok=True)

        # with open("models/preprocessor.b", "wb") as f_out:
        #     pickle.dump(dv, f_out)
        # mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(model, artifact_path="models_sklearn")

        print(f'Model intercept: {model.intercept_}')  # For Question 5
        print(f"Training RMSE: {rmse:.2f}")
        print(f"Logged to run {run.info.run_id}")

    return dict(model=model, dv=dv)
