from sklearn.linear_model import LogisticRegression


class Model(LogisticRegression):
    def __init__(self, **kwargs):
        kwargs.setdefault("solver", "liblinear")
        super().__init__(**kwargs)
