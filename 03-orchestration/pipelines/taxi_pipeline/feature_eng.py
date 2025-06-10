from sklearn.feature_extraction import DictVectorizer

@transformer
def execute(data: dict, **kwargs):
    df_train = data['df_train']
    df_val = data['df_val']

    categorical = ['PU_DO']
    numerical = ['trip_distance']

    def make_features(df, dv=None):
        dicts = df[categorical + numerical].to_dict(orient='records')
        if dv is None:
            dv = DictVectorizer()
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)
        return X, dv

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    X_train, dv = make_features(df_train)
    X_val, _ = make_features(df_val, dv)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'dv': dv
    }
