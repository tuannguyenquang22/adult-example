from ray import tune
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Search space
config = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(3, 20),
    "max_features": tune.choice(["sqrt", "log2", None]),
    "min_samples_split": tune.randint(2, 10),
}

# Objective function
def objective(config, X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        max_features=config["max_features"],
        min_samples_split=config["min_samples_split"],
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    tune.report(accuracy=accuracy)
