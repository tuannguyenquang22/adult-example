from ray import tune
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


config = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(3, 10),
    "learning_rate": tune.uniform(0.01, 0.1),
    "subsample": tune.uniform(0.5, 1.0),
}


def objective(config, X_train, X_test, y_train, y_test):
    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    tune.report(accuracy=accuracy)
