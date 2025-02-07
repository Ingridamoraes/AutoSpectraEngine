import os
import pandas as pd
import random

from deap import base, creator, tools, algorithms
from auto_spectra_engine.util import load_data, insert_results_subpath, plot_all_performances, split_spectrum_from_csv, append_results_to_csv, get_pipeline_combinations
from auto_spectra_engine.preprocessing import mean_centering, autoscaling, smoothing, first_derivative, second_derivative, msc, snv, iso_forest_outlier_removal
from auto_spectra_engine.analysis_regression import plot_PCA, get_plsr_performance 
from auto_spectra_engine.analysis_oneclass import OneClassPLS, DDSIMCA
from auto_spectra_engine.analysis_classification import get_RF_performance, get_plsda_performance

from multiprocessing import Pool

try:
    # Check if running in a Jupyter Notebook or Colab
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        from tqdm.notebook import tqdm  # Use notebook-friendly tqdm
    else:
        from tqdm import tqdm  # Use standard tqdm for console
except Exception:
    from tqdm import tqdm  # Default to console-friendly tqdm

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def run_all_experiments(file, modelo="PLSDA", coluna_predicao="Class", use_contamination=None, pipeline_family='all', use_split_spectra=None):
    pipelines = get_pipeline_combinations(pipeline_family)
    
    if use_contamination is None:
        use_contamination=[0]
    
    if use_split_spectra == 0:
        use_split_spectra = None

    result = []
    for pipeline in tqdm(pipelines, desc='Processing pipelines', leave=False):
        # Exclude combinations with "d2 + d1" and "msc + snv"
        for c in tqdm(use_contamination, desc='Processing contamination', leave=False):
            if use_split_spectra is None:
                result.append(run_experiment(file, modelo=modelo, coluna_predicao=coluna_predicao, contamination=c, combinacao=pipeline, plot=False, verbose=False))
            else:
                spectrum_intervals = split_spectrum_from_csv(file, use_split_spectra)
                for i, (start_index, end_index) in tqdm(enumerate(spectrum_intervals), desc='Processing Spectra Splits', leave=False, unit="slice",
                                        bar_format="{desc} {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]"):
                    result.append(run_experiment(file, start_index=start_index, end_index=end_index, 
                                                modelo=modelo, coluna_predicao=coluna_predicao, 
                                                contamination=c, combinacao=pipeline, 
                                                plot=False, verbose=False))

    result_df = pd.DataFrame(result, columns=["LV", "Pipeline", "Outliers (c)", "Performance", "Start Index", "End Index"])
    result_df.to_csv("ExperimentResults.csv", index=False)
    plot_all_performances(result_df)



def run_experiment(file, start_index=0, end_index=None, contamination=0.0, combinacao=None, 
                   cortar_extremidades=False, inicio_espectro_col=3, plot=True, modelo="PLSDA", 
                   inlier_class="Pure", coluna_predicao=0, verbose=True):
    
    if verbose:
        print("** AutoSpectraEngine (v0.1) ::: Run Experiment **")
        print(f"Processing file:\"{os.path.basename(file)}\"")
        if combinacao is None:
            combinacao = "mc+smo+d1"
            print("Hyperparameter \"combination\" was not used, applying standard value \"mc+smo+d1\"")

    resultados = []
    file_name_no_ext = os.path.splitext(file)[0] 
    data = load_data(file).infer_objects()

    X = data.iloc[:, inicio_espectro_col:]
    Ys = data.iloc[:, :inicio_espectro_col]

    if cortar_extremidades:
        if verbose: print("Removing extremities...")
        perc_corte = int(X.shape[1] * 0.02)
        X = X.iloc[:, perc_corte:-perc_corte]
        start_index = perc_corte

    X = X.iloc[:, start_index:] if end_index is None else X.iloc[:, start_index:end_index]
    label_espectro = X.columns.str.split('.').str[0]

    df_pp = X
    pre_processing_steps = {
        "mc": mean_centering,
        "scal": autoscaling,
        "smo": smoothing,
        "d2": second_derivative,
        "d1": first_derivative,
        "msc": msc,
        "snv": snv
    }
    
    for step, func in pre_processing_steps.items():
        if step in combinacao:
            if verbose: print(f"Applying {step.upper()}...")
            df_pp = pd.DataFrame(func(df_pp.values), columns=df_pp.columns)

    if contamination > 0:
        if verbose: print(f"Outliers removal (Isolation Forest, c ={contamination})...")
        sub_data, sub_Ys = iso_forest_outlier_removal(df_pp.values, Ys, contamination)
    else:
        sub_data, sub_Ys = df_pp, Ys

    if plot and verbose:
        print("Plotting PCA...")
        plot_PCA(sub_data.values, sub_Ys.loc[:, coluna_predicao], combinacao, insert_results_subpath(file_name_no_ext))

    perf_return = []
    model_results = []

    model_configs = {
        "PLSR": {
            "function": get_plsr_performance,
            "params": (sub_data.values, sub_Ys.loc[:, coluna_predicao], combinacao, 0.33, 10, 5, plot, file_name_no_ext),
            "columns": ["file", "cortar_extremidades", "pre_processamento", "contaminacao",
                        "best_n_components", "rmse_cal", "rmsecv", "rmse_test", "RPD", "RER",
                        "r2_cal", "r2_cv", "r2_test", "start_index", "end_index"],
            "sort_by": "rmse_test",
            "file_suffix": "best_PLSR.csv"
        },
        "PLSDA": {
            "function": get_plsda_performance,
            "params": (sub_data.values, sub_Ys.loc[:, coluna_predicao], combinacao, 0.33, 18, 10, label_espectro, plot, file_name_no_ext),
            "columns": ["file", "cortar_extremidades", "pre_processamento", "contaminacao",
                        "best_n_components", "accuracy", "start_index", "end_index"],
            "sort_by": "accuracy",
            "file_suffix": "best_PLSDA.csv"
        },
        "RF": {
            "function": get_RF_performance,
            "params": (sub_data.values, sub_Ys.loc[:, coluna_predicao], 0.33, 10, plot, label_espectro, file_name_no_ext),
            "columns": ["file", "cortar_extremidades", "pre_processamento", "contaminacao",
                        "accuracy_mean", "accuracy_std", "accuracy_min", "accuracy_max",
                        "seed_accuracy_max", "sensibility_mean", "sensibility_std",
                        "sensibility_min", "sensibility_max", "specificity_mean",
                        "specificity_std", "specificity_min", "specificity_max", "start_index", "end_index"],
            "sort_by": "accuracy_mean",
            "file_suffix": "best_RF.csv"
        },
        "OneClassPLS": {
            "function": OneClassPLS(n_components=2, inlier_class=inlier_class, n_splits=3, plotar=plot, file_name_no_ext=file_name_no_ext, coluna_y_nome=coluna_predicao).fit_and_evaluate_full_pipeline,
            "params": (sub_data, sub_Ys, coluna_predicao, plot),
            "columns": ["file", "cortar_extremidades", "pre_processamento", "contaminacao",
                        "accuracy", "best_n_components", "sensitivity", "specificity", "start_index", "end_index"],
            "sort_by": "accuracy",
            "file_suffix": "best_OneClassPLS.csv"
        },
        "DDSIMCA": {
            "function": DDSIMCA(max_n_components=2, inlier_class=inlier_class, plotar_DDSIMCA=plot).fit_and_evaluate_full_pipeline,
            "params": (sub_data, sub_Ys, coluna_predicao),
            "columns": ["file", "cortar_extremidades", "pre_processamento", "contaminacao",
                        "accuracy", "best_n_components", "sensitivity", "specificity", "start_index", "end_index"],
            "sort_by": "accuracy",
            "file_suffix": "best_DDSIMCA.csv"
        }
    }

    if modelo in model_configs:
        model = model_configs[modelo]
        if verbose:
            print(f"Executing {modelo}...")

        model_results = model["function"](*model["params"])
        rounded_model_results = [round(x, 2) if isinstance(x, (int, float)) else x for x in model_results]
        resultados.append([
            file, cortar_extremidades, combinacao, round(contamination, 2),
            *rounded_model_results, start_index, end_index
        ])
        result_file = f"{insert_results_subpath(file_name_no_ext)}_{coluna_predicao}_{model['file_suffix']}"
        append_results_to_csv(resultados, model["columns"], result_file)

        perf_metric = rounded_model_results[-1] 

    perf_return = [model_results[0], combinacao, round(contamination, 2), round(perf_metric, 2), start_index, end_index]
    return perf_return



# Define the function
def run_ga_experiments(file, modelo="PLSDA", coluna_predicao="Adulterant", pipeline_family="Raman", budget=100, use_parallelization=False):
    """
    Uses a Genetic Algorithm (GA) to find the best experimental configuration for spectral analysis.

    Parameters:
    file (str): Path to the spectral dataset.
    modelo (str): Machine learning model to evaluate (e.g., "PLSDA", "PLSR").
    coluna_predicao (str): Target column name.
    pipeline_family (str): Type of pipeline (e.g., "Raman").
    budget (int): Number of GA evaluations (default=100).

    Returns:
    dict: Best experiment configuration.
    """

    # Define the available preprocessing pipelines
    combinacao_options = get_pipeline_combinations(pipeline_family)

    # Load data to get the number of spectral columns
    data = load_data(file).infer_objects()
    num_columns = data.shape[1]

    # GA Parameter Ranges
    START_INDEX_RANGE = (0, num_columns // 2)  # Spectral start index
    END_INDEX_RANGE = (num_columns // 2, num_columns)  # Spectral end index
    CONTAMINATION_RANGE = (0.0, 0.2)  # Percentage of outliers to remove
    COMBINACAO_INDEX_RANGE = (0, len(combinacao_options) - 1)  # Preprocessing selection

    # Define the fitness function (maximize performance)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax) #TODO: Implementar quando é um problema de minimização

    def get_random_in_interval(start, end, step=10):
        return random.randrange(start, end + 1, step)

    # Individual (Solution) Definition
    def create_individual():
        start_index = get_random_in_interval(*START_INDEX_RANGE)
        min_end_index = start_index + 50
        max_end_index = END_INDEX_RANGE[1]
        if min_end_index > max_end_index:
            raise ValueError("Cannot satisfy the minimum distance constraint. Adjust the ranges.")
        end_index = get_random_in_interval(min_end_index, max_end_index)
        contamination = round(random.uniform(*CONTAMINATION_RANGE), 2)
        combinacao_index = random.randint(*COMBINACAO_INDEX_RANGE)
        return [start_index, end_index, contamination, combinacao_index]

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Parallelization
    if use_parallelization:
        pool = Pool()
        toolbox.register("map", pool.map)

    def mutate_individual(individual):
        individual[0] = get_random_in_interval(*START_INDEX_RANGE)  # start_index (multiple of 10)
        # Ensure end_index is at least 50 greater than start_index
        min_end_index = individual[0] + 50
        max_end_index = END_INDEX_RANGE[1]
        if min_end_index > max_end_index:
            raise ValueError("Cannot satisfy the minimum distance constraint during mutation. Adjust the ranges.")
        individual[1] = get_random_in_interval(min_end_index, max_end_index)  # end_index (multiple of 10)
        individual[2] = round(random.uniform(*CONTAMINATION_RANGE), 2)  # contamination (float)
        individual[3] = random.randint(*COMBINACAO_INDEX_RANGE)  # combinacao (int)
        return individual,


    # Evaluation Function
    def evaluate(individual):
        start_index, end_index, contamination, combinacao_index = individual
        combinacao = combinacao_options[combinacao_index]

        # Ensure start_index < end_index
        if end_index - start_index <= 100:
            return (0.0,)

        # Run the experiment
        result = run_experiment(file, start_index=start_index, end_index=end_index, contamination=contamination, combinacao=combinacao, modelo=modelo, coluna_predicao=coluna_predicao, verbose=False, plot=False)
        #print(f"Performance: {result[3]} (Start: {start_index}, End: {end_index}, Contamination: {contamination}, Pipeline: {combinacao})")
        # Extract performance metric (assume last value in result is the score)
        return (result[3],)  # Maximize performance

    # Register GA Operations
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)  # Crossover: Two-point crossover
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selBest)  # Selection: Keep best individuals

    # Initialize Population
    pop_size = 20  # Population size
    generations = budget // pop_size  # Number of generations
    population = toolbox.population(n=pop_size)

    with tqdm(total=generations, desc="GA Progress") as pbar:
        for gen in range(generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
            fits = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit
            population[:] = toolbox.select(offspring, k=len(population))
            pbar.update(1)

    # Extract the Best Individual
    best_individual = tools.selBest(population, k=1)[0] #Restore the three best individuals.
    best_config = {
        "start_index": best_individual[0],
        "end_index": best_individual[1],
        "contamination": best_individual[2],
        "combinacao": combinacao_options[best_individual[3]],
        "performance": best_individual.fitness.values[0]
    }

    return best_config