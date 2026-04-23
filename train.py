import os
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# create models folder
os.makedirs("models", exist_ok=True)

# load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# hyperparameter variations
n_estimators_list = [50, 100]
max_depth_list = [2, 3]

print("Training multiple models...\n")

for n in n_estimators_list:
    for depth in max_depth_list:
        
        model = RandomForestClassifier(
            n_estimators=n,
            max_depth=depth,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model -> n_estimators={n}, depth={depth}, accuracy={acc}")
        
        # save model with version name
        filename = f"models/model_n{n}_d{depth}.pkl"
        joblib.dump(model, filename)