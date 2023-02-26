import pandas as pd
import pickle

# Setting the working directory
base_path = "/home/sagemaker-user/"
import os
os.chdir(base_path)

submission = pd.read_csv("data/raw/test/test.csv")[["Id"]]
test = pd.read_csv("data/processed/test/test.csv")

# Load the saved model from disk
latest_model = sorted(os.listdir("models/lgbm"))[-1]
print(latest_model)
with open(f"models/lgbm/{latest_model}/model.bin", 'rb') as f:
    model = pickle.load(f)

pred_val = model.predict(test)

submission["quality"] = pred_val

def scale(df):
    df["quality"] = df["quality"] + 3
    return df

submission = scale(submission)

submission.to_csv('data/output/submission.csv', index=False)