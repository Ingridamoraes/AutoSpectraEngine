import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
from sklearn.utils import shuffle   
from sklearn.metrics.pairwise import euclidean_distances

def display_general_info(df):
    """
    Exibe informações gerais e estatísticas básicas do DataFrame.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.

    Retorna:
    None
    """
    print("Informações gerais:")
    print(df.info())
    print("\nEstatísticas descritivas:")
    print(df.describe())

def check_missing_data(df):
    """
    Verifica e visualiza dados ausentes no DataFrame.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.

    Retorna:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Mapa de Dados Ausentes')
    plt.show()

def plot_variable_distributions(df, columns):
    """
    Plota as distribuições das variáveis espectrais.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.
    columns (list): Lista de colunas para plotar as distribuições.

    Retorna:
    None
    """
    df[columns].hist(bins=30, figsize=(30, 20))
    plt.suptitle('Distribuições das Variáveis Espectrais')
    plt.show()

def plot_correlations(df):
    """
    Plota a matriz de correlações entre as variáveis espectrais.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.

    Retorna:
    None
    """
    plt.figure(figsize=(30, 20))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de Correlações')
    plt.show()

def plot_spectral_data(df, wavelength_columns, title='Espectros Médios e Variabilidade'):
    """
    Plota os espectros médios e variabilidade.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.
    wavelength_columns (list): Lista de colunas que representam as variáveis espectrais (comprimentos de onda).

    Retorna:
    None
    """
    mean_spectrum = df[wavelength_columns].mean()
    std_spectrum = df[wavelength_columns].std()

    plt.figure(figsize=(14, 7))
    plt.plot(wavelength_columns, mean_spectrum, label='Espectro Médio', color='blue')
    plt.fill_between(wavelength_columns, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color='blue', alpha=0.2, label='Variabilidade (±1 std)')
    plt.title(title)
    plt.xlabel('Comprimento de Onda')

    plt.xticks(np.arange(0, len(wavelength_columns)+1, 5), rotation=45, ha='right')
    plt.ylabel('Intensidade')
    plt.legend()
    plt.show()

def plot_individual_spectra(df, wavelength_columns):
    """
    Plota os espectros individuais para todas as linhas no DataFrame.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados.
    wavelength_columns (list): Lista de colunas que representam as variáveis espectrais (comprimentos de onda).

    Retorna:
    None
    """
    plt.figure(figsize=(14, 7))

    for idx in df.index:
        plt.plot(wavelength_columns, df.loc[idx, wavelength_columns], label=f'Espectro {idx}')

    plt.title('Espectros Individuais')
    plt.xlabel('Comprimento de Onda')
    plt.ylabel('Intensidade')
    #plt.legend()
    plt.show()

def kennard_stone_split(X=None, y=None, sample_id=None, test_size=0.3, random_state=None, stratify=None):
    if stratify is not None:
        unique_classes, y_indices = np.unique(stratify, return_inverse=True)
        train_indices, test_indices = [], []

        for cls in unique_classes:
            class_indices = np.where(y_indices == cls)[0]
            X_class = X[class_indices]

            #if y is not None:
            #    y_class = y[class_indices]
            #else:
            #    y_class = None

            ks_train_indices, ks_test_indices = kennard_stone_indices(X_class, test_size)

            train_indices.extend(class_indices[ks_train_indices])
            test_indices.extend(class_indices[ks_test_indices])
    else:
        train_indices, test_indices = kennard_stone_indices(X, test_size)

    # Shuffle the indices
    if random_state is not None:
        np.random.seed(random_state)

    train_indices = shuffle(train_indices, random_state=random_state)
    test_indices = shuffle(test_indices, random_state=random_state)

    # Create the train-test split
    X_train, X_test = X[train_indices], X[test_indices]
    train_label_id, test_label_id = sample_id[train_indices], sample_id[test_indices]

    if y is not None:
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test, train_label_id, test_label_id
    else:
        return X_train, X_test, train_label_id, test_label_id


def insert_results_subpath(file_path):
    """
    Modify a file path by inserting "results" as a subdirectory before the file name.

    Parameters:
        file_path (str): The original file path.

    Returns:
        str: The modified file path with "results" added as a subdirectory.
    """
    # Get the directory and file name from the original path
    dir_name, file_name = os.path.split(file_path)

    # Create the new path with the "results" subdirectory
    new_path = os.path.join(dir_name, "results", file_name.split(".")[0])

    return new_path

def kennard_stone_indices(X, test_size=0.3):
    n_samples = X.shape[0]

    if isinstance(test_size, float):
        n_test = int(np.ceil(test_size * n_samples))
    else:
        n_test = test_size

    n_train = n_samples - n_test

    # Step 1: Compute pairwise distances
    distances = euclidean_distances(X, X)

    # Step 2: Select the first point which is the one farthest from the center (mean)
    mean_point = np.mean(X, axis=0)
    initial_index = np.argmax(np.linalg.norm(X - mean_point, axis=1))

    train_indices = [initial_index]
    remaining_indices = list(range(n_samples))
    remaining_indices.remove(initial_index)

    # Step 3: Iteratively select points that are farthest from the already selected points
    for _ in range(1, n_train):
        dist_to_selected = np.min(distances[train_indices, :], axis=0)
        next_index = remaining_indices[np.argmax(dist_to_selected[remaining_indices])]
        train_indices.append(next_index)
        remaining_indices.remove(next_index)

    # The remaining points are used as the test set
    test_indices = remaining_indices

    return train_indices, test_indices

def load_data(file_path):
    """
    Carrega a base de dados espectrais a partir de um arquivo CSV.

    Parâmetros:
    file_path (str): Caminho para o arquivo CSV.

    Retorna:
    DataFrame: DataFrame contendo os dados carregados.
    """
    df = pd.read_csv(file_path)
    df = df.fillna(0)
    return df

def plot_performance_comparison(data):
    """
    Plots a bar chart to compare the performance of pipelines using the original columns.

    The plot combines "LV" and "Outliers (c)" to create a more descriptive x-axis for the pipelines.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the columns ["LV", "Outliers (c)", "Pipeline", "Performance"].
    """
    # Ensure the required columns are in the DataFrame
    required_columns = ["LV", "Outliers (c)", "Pipeline", "Performance"]
    if not set(required_columns).issubset(data.columns):
        raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")

    # Create a new column for combined LV and Outliers description
    data['Combination'] = "Pipeline: " + data['Pipeline'].astype(str) + " | LV: " + data['LV'].astype(str) + " | Outliers: " + data['Outliers (c)'].astype(str)

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=data,
        x="Pipeline",
        y="Performance",
        hue="Combination",
    )

    # Customize plot appearance
    plt.title("Performance Comparison of Pipelines", fontsize=16)
    plt.xlabel("Pipeline (Combination of LV and Outliers)", fontsize=14)
    plt.ylabel("Performance", fontsize=14)
    #plt.legend(title="Combination", title_fontsize=12, fontsize=10)
    plt.legend().remove() 
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

def split_spectrum_from_csv(file_path: str, num_splits: int):
    """
    Splits a spectrum into equal column-based intervals from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
        num_splits (int): Number of splits to be made.

    Returns:
        list of tuples: A list containing (start_index, end_index) for each slice.
    """
    # Load CSV assuming it's a single-row dataset
    df = pd.read_csv(file_path, header=None, dtype=str)
    
    # Flatten the single row into a list
    num_columns = df.shape[1]  # Get the number of columns

    # Ensure there are enough columns to split
    if num_splits > num_columns:
        raise ValueError(f"Number of splits ({num_splits}) is greater than available columns ({num_columns}).")

    # Compute split size
    step = num_columns // num_splits
    remainder = num_columns % num_splits  # Handle cases where the columns aren't perfectly divisible

    # Generate the split intervals
    intervals = []
    start_idx = 0
    for i in range(num_splits):
        end_idx = start_idx + step + (1 if i < remainder else 0)  # Distribute remainder among first splits
        remainder -= 1 if i < remainder else 0  # Reduce remainder as it's used
        intervals.append((start_idx, None if i == num_splits - 1 else end_idx))
        start_idx = end_idx  # Move to the next section
    
    return intervals

def append_results_to_csv(resultados, columns, file_name):
    """
    Appends results to a CSV file, ensuring headers are written only once.

    Parameters:
    - resultados: List of result values to be appended.
    - columns: List of column names.
    - file_name: Path to the CSV file.
    """
    file_exists = os.path.exists(file_name)  # Check if file exists
    resultados_df = pd.DataFrame(resultados, columns=columns)  # Convert to DataFrame
    resultados_df.sort_values(columns[4:], ascending=False).to_csv(
        file_name, index=False, mode='a', header=not file_exists
    )  # Append results

def plot_all_performances(df):
    """
    Plots performance comparisons using bar plots for different pipeline instances,
    considering Outliers (c) and LV.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ["LV", "Pipeline", "Outliers (c)", "Performance", "Start Index", "End Index"]
    """
    # Create a unique identifier for each pipeline instance
    df["Pipeline Instance"] = df["Pipeline"] + " (" + df["Start Index"].astype(str) + "-" + df["End Index"].astype(str) + ")"

    # Sort the DataFrame by Performance in descending order
    df_sorted = df.sort_values(by="Performance", ascending=False)

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_sorted, x="Pipeline Instance", y="Performance", hue="Start Index", palette="flare")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add labels and title
    plt.xlabel("Pipeline Instance")
    plt.ylabel("Performance")
    plt.title("Performance Comparison by Pipeline Instance")

    ax.grid(axis="y", linestyle="--", alpha=0.7)  # Dashed grid lines for clarity

    # Show the plot
    plt.show()

def get_pipeline_combinations(pipeline_family='all'):
    # TODO: @ingrid: definir quais as combinações para cada família de pipeline
    pipelines = ["mc"]
    if pipeline_family == 'all':
        pipelines = ['mc', 'scal', 'smo', 'd2', 'd1', 'msc', 'snv',
                    'mc + scal', 'mc + smo', 'mc + d2', 'mc + d1', 'mc + msc', 'mc + snv',
                    'scal + smo', 'scal + d2', 'scal + d1', 'scal + msc', 'scal + snv',
                    'smo + d2', 'smo + d1', 'smo + msc', 'smo + snv',
                    'd2 + msc', 'd2 + snv', 'd1 + msc', 'd1 + snv',
                    'mc + scal + smo', 'mc + scal + d2', 'mc + scal + d1', 'mc + scal + msc', 'mc + scal + snv',
                    'mc + smo + d2', 'mc + smo + d1', 'mc + smo + msc', 'mc + smo + snv',
                    'mc + d2 + msc', 'mc + d2 + snv', 'mc + d1 + msc', 'mc + d1 + snv',
                    'scal + smo + d2', 'scal + smo + d1', 'scal + smo + msc', 'scal + smo + snv',
                    'scal + d2 + msc', 'scal + d2 + snv', 'scal + d1 + msc', 'scal + d1 + snv',
                    'smo + d2 + msc', 'smo + d2 + snv', 'smo + d1 + msc', 'smo + d1 + snv',
                    'mc + smo + d2', 'mc + smo + d1', 'mc + smo + msc', 'mc + smo + snv',
                    'mc + d2 + msc', 'mc + d2 + snv', 'mc + d1 + msc', 'mc + d1 + snv',
                    'mc + smo + d2 + msc', 'mc + smo + d2 + snv', 'mc + smo + d1 + msc', 'mc + smo + d1 + snv',
                    'scal + smo + d2 + msc', 'scal + smo + d2 + snv', 'scal + smo + d1 + msc', 'scal + smo + d1 + snv',
                    'mc + smo + d2 + msc', 'mc + smo + d2 + snv', 'mc + smo + d1 + msc', 'mc + smo + d1 + snv'
                    ]
    elif pipeline_family == 'NIR':
        pipelines = ['mc', 'mc + smo', 'mc + d2', 'mc + d1', 'mc + msc', 'mc + snv']
    elif pipeline_family == 'Raman':
        pipelines = ['mc', 'mc + smo', 'mc + d2', 'mc + d1']
        #pipelines = ['mc']
    return pipelines
    
"""
result = [
    [18,"mc",0,0.93,0,395.0],
    [12,"mc",0,0.96,395,790.0],
    [10,"mc",0,0.67,790,1184.0],
    [11,"mc",0,0.78,1184,1578.0],
    [6,"mc",0,0.81,1578],
    [18,"mc + smo",0,0.93,0,395.0],
    [12,"mc + smo",0,0.96,395,790.0],
    [10,"mc + smo",0,0.67,790,1184.0],
    [11,"mc + smo",0,0.78,1184,1578.0],
    [6,"mc + smo",0,0.81,1578],
    [5,"mc + d2",0,0.63,0,395.0],
    [14,"mc + d2",0,0.84,395,790.0],
    [5,"mc + d2",0,0.61,790,1184.0],
    [18,"mc + d2",0,0.56,1184,1578.0],
    [2,"mc + d2",0,0.5,1578],
    [9,"mc + d1",0,0.84,0,395.0],
    [12,"mc + d1",0,0.95,395,790.0],
    [10,"mc + d1",0,0.6,790,1184.0],
    [9,"mc + d1",0,0.72,1184,1578.0],
    [6,"mc + d1",0,0.75,1578]
]
df = pd.DataFrame(result, columns=["LV", "Pipeline", "Outliers (c)", "Performance", "Start Index", "End Index"])
plot_all_performances(df)
"""