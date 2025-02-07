import numpy as np
import pandas as pd

from scipy.signal import savgol_filter 
from sklearn.ensemble import IsolationForest

def msc(input_data):
    """
    Aplica a correção de espalhamento multiplicativo (MSC) aos dados espectrais.

    Parâmetros:
    input_data (numpy.ndarray): Dados espectrais de entrada.

    Retorna:
    numpy.ndarray: Dados espectrais corrigidos por MSC.
    """
    # Calcular o espectro médio
    mean_spectrum = np.mean(input_data, axis=0)

    # Inicializar a matriz de dados corrigidos
    corrected_data = np.zeros_like(input_data)

    # Aplicar MSC para cada espectro
    for i in range(input_data.shape[0]):
        fit = np.polyfit(mean_spectrum, input_data[i, :], 1, full=True)
        corrected_data[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]

    return corrected_data

def snv(input_data):
    """
    Aplica a normalização SNV (Standard Normal Variate) aos dados espectrais.

    Parâmetros:
    input_data (numpy.ndarray): Dados espectrais de entrada.

    Retorna:
    numpy.ndarray: Dados espectrais normalizados por SNV.
    """
    # Subtrair a média e dividir pelo desvio padrão para cada espectro
    return (input_data - np.mean(input_data, axis=1, keepdims=True)) / np.std(input_data, axis=1, keepdims=True)

def first_derivative(input_data, window_size=7, poly_order=2):
    """
    Calcula a primeira derivada dos dados espectrais.

    Parâmetros:
    input_data (numpy.ndarray): Dados espectrais de entrada.
    window_size (int): Tamanho da janela do filtro de Savitzky-Golay.
    poly_order (int): Ordem do polinômio do filtro de Savitzky-Golay.

    Retorna:
    numpy.ndarray: Primeira derivada dos dados espectrais.
    """
    return savgol_filter(input_data, window_length=window_size, polyorder=poly_order, deriv=1)

def second_derivative(input_data, window_size=7, poly_order=2):
    """
    Calcula a segunda derivada dos dados espectrais.

    Parâmetros:
    input_data (numpy.ndarray): Dados espectrais de entrada.
    window_size (int): Tamanho da janela do filtro de Savitzky-Golay.
    poly_order (int): Ordem do polinômio do filtro de Savitzky-Golay.

    Retorna:
    numpy.ndarray: Segunda derivada dos dados espectrais.
    """
    return savgol_filter(input_data, window_length=window_size, polyorder=poly_order, deriv=2)

def smoothing(input_data, window_size=3, poly_order=2):
    """
    Aplica a suavização aos dados espectrais usando o filtro de Savitzky-Golay.

    Parâmetros:
    input_data (numpy.ndarray): Dados espectrais de entrada.
    window_size (int): Tamanho da janela do filtro de Savitzky-Golay.
    poly_order (int): Ordem do polinômio do filtro de Savitzky-Golay.

    Retorna:
    numpy.ndarray: Dados espectrais suavizados.
    """
    return savgol_filter(input_data, window_length=window_size, polyorder=poly_order)

def mean_centering(input_data):
    """
    Aplica a centralização pela média aos dados espectrais.

    Parâmetros:
    input_data (numpy.ndarray): Dados espectrais de entrada.

    Retorna:
    numpy.ndarray: Dados espectrais centralizados pela média.
    """
    # Calcular a média de cada variável (coluna)
    mean_values = np.mean(input_data, axis=0)

    # Subtrair a média de cada variável
    centered_data = input_data - mean_values

    return centered_data

def autoscaling(input_data):
    """
    Aplica o autoscaling (normalização por z-score) aos dados espectrais.

    Parâmetros:
    input_data (numpy.ndarray): Dados espectrais de entrada.

    Retorna:
    numpy.ndarray: Dados espectrais após o autoscaling.
    """
    # Calcular a média e o desvio padrão de cada variável (coluna)
    mean_values = np.mean(input_data, axis=0)
    std_values = np.std(input_data, axis=0)

    # Subtrair a média e dividir pelo desvio padrão de cada variável
    scaled_data = (input_data - mean_values) / std_values

    return scaled_data

def iso_forest_outlier_removal(X, Ys, contamination):
     # Convert data to DataFrame if it is not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if contamination<=0:
      return X, Ys

    # Select only numerical columns
    numerical_columns = X.select_dtypes(include='number').columns
    numerical_data = X[numerical_columns]

    # Check if there are numerical columns
    if numerical_data.empty:
        raise ValueError("No numerical columns found in the input data.")

    # Initialize the Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)

    # Fit the model and predict outliers
    outliers = iso_forest.fit_predict(numerical_data)

    # Filter out the outliers
    clean_data = X[outliers != -1]
    clean_targets = Ys[outliers != -1]

    #print('cont x',clean_data.shape) #apenas para checar
    #print('cont y', clean_targets.shape) #apenas para checar

    return clean_data, clean_targets


def iso_forest_outlier_removal(X, Ys, contamination):
     # Convert data to DataFrame if it is not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if contamination<=0:
      return X, Ys

    # Select only numerical columns
    numerical_columns = X.select_dtypes(include='number').columns
    numerical_data = X[numerical_columns]

    # Check if there are numerical columns
    if numerical_data.empty:
        raise ValueError("No numerical columns found in the input data.")

    # Initialize the Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)

    # Fit the model and predict outliers
    outliers = iso_forest.fit_predict(numerical_data)

    # Filter out the outliers
    clean_data = X[outliers != -1]
    clean_targets = Ys[outliers != -1]

    #print('cont x',clean_data.shape) #apenas para checar
    #print('cont y', clean_targets.shape) #apenas para checar

    return clean_data, clean_targets