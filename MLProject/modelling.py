import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
x_train = pd.read_csv("heart_preprocessing/x_train.csv")
x_test = pd.read_csv("heart_preprocessing/x_test.csv")
y_train = pd.read_csv("heart_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("heart_preprocessing/y_test.csv").values.ravel()

mlflow.set_experiment("CI_CD_Docker")

# training
with mlflow.start_run():
    mlflow.autolog()
    
    model = RandomForestClassifier(random_state=7)
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.4f}")

# save model
model_path = "saved_model"
mlflow.sklearn.save_model(model, model_path)
print("Training done. Model disimpen di %s" % model_path)