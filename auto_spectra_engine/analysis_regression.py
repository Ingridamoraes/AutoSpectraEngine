import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error, mean_squared_error, r2_score

def plot_PCA(X, Y, combinacao, file_name_no_ext):
    # Perform PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    # Get explained variance ratios
    explained_variance = pca.explained_variance_ratio_

    # Plot using matplotlib
    plt.figure(figsize=(10, 8))

    # Check if Y is numeric or categorical
    if np.issubdtype(Y.dtype, np.number):
        # Y is numeric (continuous values)
        scatter = plt.scatter(components[:, 0], components[:, 1], c=Y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Target Variable (Numeric)')
    else:
        # Y is categorical (labels)
        categories = Y.unique()
        colors = plt.cm.get_cmap('viridis', len(categories))
        for i, category in enumerate(categories):
            plt.scatter(components[Y == category, 0], components[Y == category, 1], color=colors(i), label=str(category))
        plt.legend(title='Target Variable (Categorical)')

    # Set labels and title with variance explained
    plt.xlabel(f'PC1 Variance Explained: {explained_variance[0]*100:.2f}%')
    plt.ylabel(f'PC2 Variance Explained: {explained_variance[1]*100:.2f}%')
    plt.title('PCA Plot:'+combinacao)
    plt.savefig(f'{file_name_no_ext}_{combinacao}_PCA.png', format='png')
    plt.show()
    plt.close()


def get_plsr_performance(X, y, preprocess_name, test_size=0.30, max_components=10, n_splits=10, plotar_plsr=True, file_name_no_ext=None):
    # Check for NaN values in X and y
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        nan_indices_X = X.isnull().any(axis=1)
    else:
        nan_indices_X = np.any(np.isnan(X), axis=1)

    nan_indices_y = np.isnan(y)
    nan_indices = np.logical_or(nan_indices_X, nan_indices_y)

    #print('nan_indices', nan_indices.sum()) #apenas para checar
    # Clear data
    X_clean = X[~nan_indices]
    y_clean = y[~nan_indices].values

    # Check if any NaN values were removed
    if np.sum(nan_indices) > 0 and plotar_plsr:
        print(f'Removed {np.sum(nan_indices)} rows with NaN values from X and y.')

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=42)

    # Restaurando os índices
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Inicializa variáveis para armazenar RMSECV, RPD e RER por número de componentes
    rmsecv = []
    r2_cv = []

    # Avalia a performance para cada número de componentes
    for n_components in range(1, max_components + 1):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_rmse = []
        fold_r2 = []
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            plsr = PLSRegression(n_components=n_components)
            plsr.fit(X_train_fold, y_train_fold)
            y_pred = plsr.predict(X_val_fold)
            fold_rmse.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
            fold_r2.append(r2_score(y_val_fold, y_pred))  # Calculate R2 score and store

        rmsecv.append(np.mean(fold_rmse))
        r2_cv.append(np.mean(fold_r2))

    # Seleciona o número de componentes com menor RMSECV
    best_n_components = np.argmin(rmsecv) + 1

    if plotar_plsr:
      plt.figure(figsize=(8, 6))
      plt.plot(range(1, max_components + 1), rmsecv, marker='o', linestyle='-', color='blue', label='RMSECV')
      plt.axvline(x=best_n_components, color='red', linestyle='--', label=f'Best LV: {best_n_components}')
      plt.xlabel('Number of Components', fontsize=15)
      plt.ylabel('RMSECV', fontsize=15)
      plt.title(f'{file_name_no_ext}-{y.name} - RMSECV by LV - {preprocess_name}')
      plt.legend(fontsize=14)
      plt.xticks(fontsize=14)  # Ajusta o tamanho da fonte dos valores do eixo x
      plt.yticks(fontsize=14)
      # Salvando a figura no Google Drive
      plt.savefig(f'{file_name_no_ext}_{y.name}_RMSECV by LV_{preprocess_name}_PLSR.png', format='png')
      plt.show()
      plt.close()

    rmsecv = rmsecv[best_n_components - 1]
    r2_cv = r2_cv[best_n_components - 1]

    plsr_best = PLSRegression(n_components=best_n_components)
    plsr_best.fit(X_train, y_train)
    y_pred_cal = plsr_best.predict(X_train)

    r2_cal = r2_score(y_train, y_pred_cal)
    rmse_cal = np.sqrt(mean_squared_error(y_train, y_pred_cal))

    # Avalia a performance no conjunto de teste
    y_pred_test = plsr_best.predict(X_test)

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    std_dev = np.std(y_clean)
    rpd = std_dev / rmse_test

    range_observed = np.max(y_clean) - np.min(y_clean)
    rer = range_observed / rmse_test

    if plotar_plsr:
        # Impressão dos resultados
        #print(f'{preprocess_name} - R² (Calibração): {r2_cal:.2f}')
        #print(f'{preprocess_name} - RMSE (Calibração): {rmse_cal:.2f}')
        #print(f'{preprocess_name} - R² (Teste): {r2_test:.2f}')
        #print(f'{preprocess_name} - RMSE (Teste): {rmse_test:.2f}')
        #print(f'{preprocess_name} - R² CV (Escolhido): {r2_cv:.2f}')
        #print(f'{preprocess_name} - RMSE CV (Escolhido): {rmsecv:.2f}')

        # Gráfico de valores medidos versus valores preditos no conjunto de teste
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred_test, color='blue', label='Predicted value')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='1:1')
        plt.xlabel('True',  fontsize=18)
        plt.ylabel('Predicted',  fontsize=18)
        plt.title(f'PLSR-{file_name_no_ext} - {y.name} - RPD={rpd:.2f}, RER={rer:.2f}, RMSE={rmse_test:.2f}, R²={r2_test:.2f}',  fontsize=10)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)  # Ajusta o tamanho da fonte dos valores do eixo x
        plt.yticks(fontsize=18)  # Ajusta o tamanho da fonte dos valores do eixo y
        # Salvando a figura no Google Drive
        plt.savefig(f'{file_name_no_ext}_{y.name}_RPD={rpd:.2f}_RER={rer:.2f}_RMSE={rmse_test:.2f}_R²={r2_test:.2f}.png', format='png')
        plt.show()
        plt.close()


    return best_n_components, rmse_cal, rmsecv, rmse_test, rpd, rer , r2_cal, r2_cv, r2_test