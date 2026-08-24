"""Week 1 In-Class Activity: Student Starter Script"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def agent_action(environment):
    if environment == "Sunny":
        return "Go Outside"
    elif environment == "Rainy":
        return "Use Umbrella"
    elif environment == "Cloudy":
        return "Stay Inside"
    elif environment == "Windy":
        return "Use Wind Protection"
    elif environment == "Snowy":
        return "Wear Winter Gear"
    elif environment == "Stormy":
        return "Stay Inside"
    return "Request Human Decision"

candidates=[Path.cwd()/"datasets",Path.cwd(),Path("/content/datasets"),Path("/content")]
data_dir=next((p for p in candidates if (p/"week1_intelligent_agents_large_dataset.csv").exists()),None)
if data_dir is None:
    raise FileNotFoundError("Upload the CSV or datasets folder.")
df=pd.read_csv(data_dir/"week1_intelligent_agents_large_dataset.csv")

print(df.shape)
print(df.head())

# TODO 1: inspect missing values
# TODO 2: apply agent_action to Environment
# TODO 3: compare with expected_action
# TODO 4: calculate accuracy
# TODO 5: create a visualization
