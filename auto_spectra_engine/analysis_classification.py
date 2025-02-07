from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

from auto_spectra_engine.util import insert_results_subpath

def get_RF_performance(X, y, test_size=0.33, n_runs=10, plotar_RF=False, feature_names=None,  file_name_no_ext=None):
    x_scale=5

    # Verificação de dados ausentes em X
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        nan_indices_X = X.isnull().any(axis=1)  # Identifica linhas com qualquer NaN em X
    else:
        nan_indices_X = np.any(np.isnan(X), axis=1)

    # Considere '0', strings vazias e NaN como rótulos inválidos
    nan_indices_y = pd.isnull(y) | (y == 0) | (y == "")  # Exclui valores 0, strings vazias e NaNs

    # Identificar amostras que contêm NaN ou valores inválidos tanto em X quanto em y
    nan_indices = np.logical_or(nan_indices_X, nan_indices_y)

    # Remover amostras com NaN ou valores inválidos em X ou em y
    X_clean = X[~nan_indices]
    y_clean = y[~nan_indices]

    # Verificar se ainda existem dados após remover as amostras inválidas
    if len(X_clean) == 0 or len(y_clean) == 0:
        print("Nenhuma amostra válida após remover as linhas com NaN ou valores inválidos.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Verificação extra: garantir que y_clean só contenha valores válidos
    unique_labels_before = set(y_clean)
    #print(f"Rótulos únicos antes da codificação: {unique_labels_before}")

    # Garantir que não exista nenhum valor que não deveria estar no vetor de rótulos
    y_clean = [label for label in y_clean if pd.notnull(label) and label != 0 and label != ""]  # Remover qualquer valor indesejado

    # Certifique-se de que os valores de y_clean são strings ou números
    y_clean = [str(label) for label in y_clean]  # Converter todos os rótulos para strings

    # Verificar se após essa limpeza ainda existem problemas
    unique_labels_after = set(y_clean)
    #print(f"Rótulos únicos após a conversão para strings: {unique_labels_after}")

    # Encode labels using LabelEncoder
    le = LabelEncoder()
    y_clean = le.fit_transform(y_clean)  # Codificar os rótulos restantes

    # Verificar e classificar os rótulos originais
    try:
      sorted_labels = sorted(le.classes_, key=lambda x: float(x))
    except ValueError:
      sorted_labels = sorted(le.classes_)
    sorted_indices = np.argsort([le.transform([label])[0] for label in sorted_labels])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=42, stratify=y_clean)
    except ValueError as e:
        print(f"Erro durante a separação treino-teste: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    accuracies = []
    sensitivities = []
    specificities = []
    confusion_matrices = []
    classification_reports = []
    seeds = list(range(n_runs))
    rf_models = []

    # Loop para realizar múltiplas execuções com diferentes seeds
    for seed in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=seed)
        rf = RandomForestClassifier(random_state=seed)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # Avaliação de performance
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred, average='macro')
        specificity = recall_score(y_test, y_pred, average='macro', labels=np.unique(y_test))

        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
        classification_reports.append(classification_report(y_test, y_pred, output_dict=True))
        rf_models.append(rf)

    # Cálculo de estatísticas de performance
    accuracy_mean = np.mean(accuracies)
    sensitivity_mean = np.mean(sensitivities)
    specificity_mean = np.mean(specificities)

    accuracy_std = np.std(accuracies)
    sensitivity_std = np.std(sensitivities)
    specificity_std = np.std(specificities)

    accuracy_min = np.min(accuracies)
    sensitivity_min = np.min(sensitivities)
    specificity_min = np.min(specificities)

    accuracy_max = np.max(accuracies)
    sensitivity_max = np.max(sensitivities)
    specificity_max = np.max(specificities)

    seed_accuracy_max = seeds[accuracies.index(accuracy_max)]
    confusion_matrix_max = confusion_matrices[accuracies.index(accuracy_max)]
    classification_report_max = classification_reports[accuracies.index(accuracy_max)]
    rf_model_max = rf_models[accuracies.index(accuracy_max)]

    if plotar_RF:
      print(accuracy_max)
      report_df = pd.DataFrame(classification_report_max).transpose()
      report_df.to_csv(f'{file_name_no_ext}_{y.name}_RF_report_df.csv')
      print(report_df.to_string())

      plt.figure(figsize=(8, 6))
      sns.heatmap(confusion_matrix_max, annot=True, fmt='d', cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)
      plt.title(f'RF {y.name} Accuracy={accuracy_max:.2f}', fontsize=24)
      plt.xlabel('Predicted', fontsize=14)
      plt.ylabel('True', fontsize=14)
      plt.xticks(fontsize=12)
      plt.yticks(fontsize=12)
      plt.savefig(f'{file_name_no_ext}_{y.name}_Accuracy={accuracy_max:.2f}_RF_', format='png')
      plt.show()
      plt.close()

      if feature_names is None:
          feature_names = X_clean.columns if isinstance(X_clean, pd.DataFrame) else range(X_clean.shape[1])

      importances = rf_model_max.feature_importances_

      # Defina os tamanhos de fonte como variáveis
      font_size_labels = 14
      font_size_ticks = 14
      font_size_y_values = 14  # Tamanho da fonte para os valores do eixo y

      fig, ax1 = plt.subplots(figsize=(10, 6))
      x_frame = pd.DataFrame(X_clean).melt()
      sns.lineplot(x='variable', y='value', data=x_frame, ax=ax1, label='Mean spectrum')

      ax2 = ax1.twinx()
      colors = plt.cm.viridis(importances / max(importances))
      x_ticks = np.arange(0, len(feature_names), x_scale)#X_acale é o intervalo de X
      x_ticklabels = [feature_names[i] for i in x_ticks]

      sns.barplot(x=feature_names, y=importances, alpha=0.7, ax=ax2, palette=colors.tolist(), hue=feature_names)

      ax1.set_xlabel('Wavenumbers (cm-1)', fontsize=font_size_labels) # se for cm-1
      #ax1.set_xlabel('Wavelengths (nm)', fontsize=font_size_labels) # se for nm
      ax1.set_ylabel('Absorbance', color='k', fontsize=font_size_labels)
      ax2.set_ylabel('Feature Importance', color='k', fontsize=font_size_labels)
      ax1.set_xticks(x_ticks)
      ax1.set_xticklabels(x_ticklabels, rotation=45, fontsize=font_size_ticks)

      # Ajustar o tamanho da fonte dos valores no eixo y do ax2
      ax1.tick_params(axis='y', labelsize=font_size_y_values)  # Ajustar para Absorbance
      ax2.tick_params(axis='y', labelsize=font_size_y_values)  # Ajustar para Feature Importance


      fig.tight_layout()
      plt.savefig(f'{insert_results_subpath(file_name_no_ext)}_{y.name}_RF_feature_importances.png', format='png')
      plt.show()

    return accuracy_mean, accuracy_std, accuracy_min, accuracy_max, seed_accuracy_max, sensitivity_mean, sensitivity_std, sensitivity_min, sensitivity_max, specificity_mean, specificity_std, specificity_min, specificity_max

def get_plsda_performance(X, y, preprocess_name, test_size=0.30, max_components=15, n_splits=10, label_espectro=[], plotar_plsda=False, file_name_no_ext="experimento_"):
    """
    Evaluate the performance of a PLS-DA model.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    preprocess_name (str): Name of the preprocessing method to be applied.
    test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.30.
    max_components (int, optional): Maximum number of PLS components to consider. Default is 20.
    n_splits (int, optional): Number of splits for cross-validation. Default is 10.
    label_espectro (list, optional): List of labels for the spectrum. Default is an empty list.
    plotar_plsda (bool, optional): Whether to plot the PLS-DA results. Default is False.
    x_scale (int, optional): Interval for the x-axis ticks in the plots. Default is 20.
    file_name_no_ext (str, optional): Base name for saving the output files. Default is "experimento_".
    """
    x_scale=20

    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        nan_indices_X = X.isnull().any(axis=1)  # Identifica linhas com qualquer NaN em X
    else:
        nan_indices_X = np.any(np.isnan(X), axis=1)

    # Considere '0', strings vazias e NaN como rótulos inválidos
    nan_indices_y = pd.isnull(y) | (y == 0) | (y == "")  # Exclui valores 0, strings vazias e NaNs

    # Identificar amostras que contêm NaN ou valores inválidos tanto em X quanto em y
    nan_indices = np.logical_or(nan_indices_X, nan_indices_y)

    # Remover amostras com NaN ou valores inválidos em X ou em y
    X_clean = X[~nan_indices]
    y_clean = y[~nan_indices]

    # Verificar se ainda existem dados após remover as amostras inválidas
    if len(X_clean) == 0 or len(y_clean) == 0:
        print("Nenhuma amostra válida após remover as linhas com NaN ou valores inválidos.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Garantir que não exista nenhum valor que não deveria estar no vetor de rótulos
    y_clean = [label for label in y_clean if pd.notnull(label) and label != 0 and label != ""]  # Remover qualquer valor indesejado

    # Certifique-se de que os valores de y_clean são strings ou números
    y_clean = [str(label) for label in y_clean]  # Converter todos os rótulos para strings

    # Encode labels using LabelEncoder
    le = LabelEncoder()
    y_clean = le.fit_transform(y_clean)  # Codificar os rótulos restantes

    # Verificar e classificar os rótulos originais
    try:
      sorted_labels = sorted(le.classes_, key=lambda x: float(x))
    except ValueError:
      sorted_labels = sorted(le.classes_)

    # Divida os dados em conjuntos de treino e teste
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=42, stratify=y_clean)
    except ValueError as e:
        print(f"Erro durante a separação treino-teste: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    ohe = OneHotEncoder(sparse_output=False)
    y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))

    accuracy = []
    rmsecv = []

    for n_components in range(1, max_components + 1):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_rmse = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train_ohe[train_index], y_train[val_index]
            plsda = PLSRegression(n_components=n_components)
            plsda.fit(X_train_fold, y_train_fold)
            y_pred = plsda.predict(X_val_fold)
            y_pred_class = np.argmax(y_pred, axis=1)

            fold_rmse.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_class)))

        rmsecv.append(np.mean(fold_rmse))


    best_n_components = np.argmin(rmsecv) + 1

    plsda_best = PLSRegression(n_components=best_n_components, scale=False)
    plsda_best.fit(X_train, y_train_ohe)
    y_pred_test = plsda_best.predict(X_test)
    y_pred_test_class = np.argmax(y_pred_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred_test_class)

    if plotar_plsda:
        # Gráfico de RMSECV por LV
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_components + 1), rmsecv, marker='o', linestyle='-', color='blue', label='RMSECV')
        plt.axvline(x=best_n_components, color='red', linestyle='--', label=f'Best LV: {best_n_components}')
        plt.xlabel('Number of Components', fontsize=15)
        plt.ylabel('RMSECV', fontsize=15)
        plt.title(f'{file_name_no_ext}-{y.name} - RMSECV by LV - {preprocess_name}', fontsize=16)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f'{insert_results_subpath(file_name_no_ext)}-{y.name} - RMSECV by LV - {preprocess_name}_PLSDA.png', format='png')
        plt.show()
        plt.close()

        # Gráfico Confusion matriz
        cm = confusion_matrix(y_test, y_pred_test_class)
        report = classification_report(y_test, y_pred_test_class, target_names=sorted_labels, output_dict=True)

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'{insert_results_subpath(file_name_no_ext)}_{y.name}_report_df_PLSDA.csv')
        #print(report_df.round(2).to_string())

        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels, annot_kws={"size": 14})

        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=14)

        plt.title(f'{file_name_no_ext} - {y.name} - Accuracy={accuracy:.2f}', fontsize=16)
        plt.xlabel('Predicted', fontsize=15, labelpad=10)
        plt.ylabel('True', fontsize=15, labelpad=10)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f'{insert_results_subpath(file_name_no_ext)}_{y.name}_Accuracy={accuracy:.2f}_PLSDA.png', format='png')
        plt.show()
        plt.close()

        # Calcula VIP scores para o modelo PLSDA por classe
        T = plsda_best.x_scores_
        W = plsda_best.x_weights_
        Q = plsda_best.y_loadings_

        # VIP para cada classe
        p, h = X_train.shape[1], best_n_components
        VIP_scores = np.zeros((p, len(sorted_labels)))  # Matriz de VIP para cada classe

        for class_idx in range(len(sorted_labels)):
            for i in range(p):
                s = np.sum([T[:, comp]**2 * W[i, comp]**2 * Q[class_idx, comp]**2 for comp in range(h)], axis=0)
                VIP_scores[i, class_idx] = np.sqrt(p * np.sum(s) / np.sum(T**2))

        # Define os ticks e os labels em escala 10^3 e ordem decrescente
        x_ticks = np.arange(0, len(label_espectro), x_scale)  # X_scale é o intervalo de X
        #x_ticklabels = [round((label_espectro[i]) / 1000, 1) for i in x_ticks]  # Divide cada valor por 1000 para a escala
        x_ticklabels=[label_espectro[i] for i in x_ticks] #caso seja um roto sem numero

        # Gerar gráficos de VIP score para cada classe
        for class_idx, class_name in enumerate(sorted_labels):
            plt.figure(figsize=(12, 12))
            plt.bar(range(1, VIP_scores.shape[0] + 1), VIP_scores[:, class_idx], color='blue')
            plt.xlabel('Wavenumbers (cm-1)', fontsize=15)
            plt.ylabel('VIP Score', fontsize=20)
            plt.title(f'VIP Scores for {class_name}', fontsize=20)

            # Usa os labels em ordem decrescente e escala 10^3
            plt.xticks(x_ticks, x_ticklabels, rotation=45, fontsize=18)
            plt.yticks(fontsize=20)

            # Salva a figura
            plt.savefig(f'{insert_results_subpath(file_name_no_ext)}_{class_name}_PLSDA_VIP.png', format='png')
            plt.show()
            plt.close()

    return best_n_components, accuracy
