from xgboost import XGBClassifier


class CustomXGBClassifier(XGBClassifier):
    def __init__(self, **kwargs):
        kwargs.setdefault('enable_categorical', True)
        kwargs.setdefault('tree_method', 'hist')
        super().__init__(**kwargs)
        
        
class Model(CustomXGBClassifier):
    pass
