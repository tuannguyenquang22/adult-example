from xgboost import XGBClassifier


class CustomXGBClassifier(XGBClassifier):
    def __init__(self, **kwargs):
        super().__init__(n_estimators=150, **kwargs)


class Model(CustomXGBClassifier):
    pass
