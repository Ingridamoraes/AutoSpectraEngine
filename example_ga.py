import auto_spectra_engine.experiment as ase
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file = "/home/barbon/Python/AutoSpectraEngine/auto_spectra_engine/datasets/NIR1.csv"
best_experiment = ase.run_ga_experiments(file, modelo="PLSR", coluna_predicao="%", budget=2000, pipeline_family="NIR", use_parallelization=True)

#best_experiment = {'start_index': 26, 'end_index': 1084, 'contamination': 0.02, 'combinacao': 'mc + d1', 'performance': 0.96}
#print(best_experiment)
#ase.run_experiment(file, start_index=best_experiment["start_index"], end_index=best_experiment["end_index"], contamination=best_experiment["contamination"], combinacao=best_experiment["combinacao"], 
#                   plot=True, modelo="PLSDA", verbose=True, coluna_predicao="Adulterant")


#result = pd.read_csv("/home/barbon/Python/AutoSpectraEngine/auto_spectra_engine/datasets/results/NIR1_Adulterant_PLSDA_Best_.csv")
#result = pd.read_csv("/home/barbon/Python/AutoSpectraEngine/auto_spectra_engine/datasets/results/NIR1_Class_best_PLSR_.csv")

column = "rmse_test"
file_results  = 'NIR1_%_best_PLSR'
result = pd.read_csv("/home/barbon/Python/AutoSpectraEngine/auto_spectra_engine/datasets/results/"+file_results+".csv")
df = result

# Set seaborn style for beautiful plots
sns.set(style="whitegrid")

heatmap_data = df.pivot_table(index="start_index", columns="end_index", values=column, aggfunc="mean")

# Improve heatmap visualization by removing the gray grid background
plt.figure(figsize=(20, 16))

# Set a white background to remove the grid effect
sns.set_style("white")

# Create a refined heatmap with better contrast and labels
sns.heatmap(
    heatmap_data,
    #cmap="coolwarm",
    cmap="YlOrBr",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    linecolor='black',
    cbar_kws={'label': column},
    xticklabels=5,
    yticklabels=5,
    square=True,
    annot_kws={"size": 9}
)

plt.title(file_results+" heatmap based on Start and End Index", fontsize=12, fontweight='bold')
plt.xlabel("End Index", fontsize=12)
plt.ylabel("Start Index", fontsize=12)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Display the improved heatmap
plt.show()
