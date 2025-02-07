import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix 
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

import seaborn as sns
from scipy.stats import chi2


class DDSIMCA:
    def __init__(self, inlier_class, max_n_components, alpha=0.05, plotar_DDSIMCA=False, file_name_no_ext="FileDDSIMCA"):
        self.max_n_components = max_n_components
        self.pca = PCA(n_components=max_n_components)
        self.scaler = StandardScaler()
        self.alpha = alpha  # Significance level for the Hotelling's T² statistic
        self.threshold_T2 = None
        self.threshold_Q = None
        self.inlier_class = inlier_class
        self.plotar_DDSIMCA = plotar_DDSIMCA
        self.file_name_no_ext = file_name_no_ext

    def fit(self, X, y):
        X_inliers = X[y == self.inlier_class]  # Filtrar dados pertencentes à classe inlier
        X_scaled = self.scaler.fit_transform(X_inliers)  # Padronizar os dados
        self.pca.fit(X_scaled)  # Ajuste do modelo PCA

        T_scores = self.pca.transform(X_scaled)  # Projetar os dados no espaço latente do PCA

        if self.n_components is None:
            self.n_components = np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= (1 - self.alpha)) + 1

        T2 = np.sum((T_scores ** 2) / np.var(T_scores, axis=0), axis=1)
        self.threshold_T2 = chi2.ppf(1 - self.alpha, df=self.n_components)

        X_reconstructed = self.pca.inverse_transform(T_scores)
        residuals = X_scaled - X_reconstructed
        Q = np.sum(residuals ** 2, axis=1)
        self.threshold_Q = np.percentile(Q, 100 * (1 - self.alpha))
        return self.threshold_T2, self.threshold_Q

    def predict(self, X):
        X_scaled = self.scaler.transform(X)  # Padronizar os dados
        T_scores = self.pca.transform(X_scaled)
        T2 = np.sum((T_scores ** 2) / np.var(T_scores, axis=0), axis=1)

        X_reconstructed = self.pca.inverse_transform(T_scores)
        residuals = X_scaled - X_reconstructed
        Q = np.sum(residuals ** 2, axis=1)

        predictions = np.where((T2 <= self.threshold_T2) & (Q <= self.threshold_Q), 1, -1)
        return predictions, T2, Q

    def plot_acceptance(self, T2, Q, y_test, predictions):
        # Transformação logarítmica para visualização
        true_labels = np.where(y_test == self.inlier_class, 1, -1)
        h_log_test = np.log(1 + T2 / self.threshold_T2)
        Q_log_test = np.log(1 + Q / self.threshold_Q)

        # Definindo a linha de fronteira
        boundary_h = np.log(1 + 1)  # log(1 + T2 / threshold_T2) onde T2 = threshold_T2
        boundary_Q = np.log(1 + 1)  # log(1 + Q / threshold_Q) onde Q = threshold_Q

        # Plot dos dados de teste com a fronteira de decisão
        plt.figure(figsize=(8, 6))

        # Máscaras para inliers, outliers e amostras incorretamente classificadas
        inliers_mask = (predictions == 1) & (true_labels == 1)       # Inliers corretamente classificados
        outliers_mask = (predictions == -1) & (true_labels == -1)    # Outliers corretamente classificados
        misclassified_mask = (predictions != true_labels)            # Amostras classificadas incorretamente

        # Plot das amostras
        plt.scatter(h_log_test[inliers_mask], Q_log_test[inliers_mask], color='#00FF00', edgecolor='black', label='Pure', alpha=1, s=50, linewidth=1)
        plt.scatter(h_log_test[outliers_mask], Q_log_test[outliers_mask], color='blue', edgecolor='black', label='Adulterated', alpha=1, s=50, linewidth=1)
        plt.scatter(h_log_test[misclassified_mask], Q_log_test[misclassified_mask], facecolors='none', edgecolor='red', label='Missclassified', alpha=1, s=50, linewidth=1)

        # Adicionando a linha de fronteira
        plt.axhline(boundary_Q, color='black', linestyle='--', linewidth=1, label="Acceptance Boundary (Q)")
        plt.axvline(boundary_h, color='black', linestyle='--', linewidth=1, label="Acceptance Boundary (T²)")

        # Customizações do gráfico
        plt.xlabel("log(1 + T²/T²₀)", fontsize=15)
        plt.ylabel("log(1 + Q/Q₀)", fontsize=15)
        plt.legend()
        plt.title("Acceptance Plot - Test Data", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{self.file_name_no_ext}_acceptance_plot_DDSIMCA.tiff", format='tiff')
        plt.show()
        plt.close()  # Fecha a figura após salvar

    def fit_and_evaluate_full_pipeline(self, df_pp, sub_Ys, coluna_predicao):
        # Preparação dos dados
        sub_Ys = sub_Ys.reset_index(drop=True)
        df_pp = df_pp.reset_index(drop=True)

        pure_mask = sub_Ys.loc[:, coluna_predicao].str.replace(" ", "") == self.inlier_class.replace(" ", "")
        non_pure_mask = ~pure_mask

        # Dados inliers
        X_inliers = df_pp[pure_mask].values
        y_inliers = sub_Ys.loc[pure_mask, coluna_predicao].values

        # Dados outliers
        X_outliers = df_pp[non_pure_mask].values
        y_outliers = sub_Ys.loc[non_pure_mask, coluna_predicao].values

        # Dividir inliers em 80% para treino e 20% para teste
        X_train_pure, X_test_pure, y_train_pure, y_test_pure = train_test_split(
            X_inliers, y_inliers, test_size=0.2, random_state=42, shuffle=True
        )

        X_test = np.vstack([X_test_pure, X_outliers])
        y_test = np.hstack([y_test_pure, y_outliers])

        best_sensitivity = 0
        best_n_components = 2
        best_accuracy = 0
        best_specificity = 0

        for n_components in range(1, self.max_n_components+1):
            self.n_components = n_components
            self.pca = PCA(n_components=n_components)

            self.fit(X_train_pure, y_train_pure)
            predictions, T2, Q = self.predict(X_test)

            true_labels = np.where(y_test == self.inlier_class, 1, -1)
            current_accuracy = np.mean(predictions == true_labels)

            cm = confusion_matrix(true_labels, predictions, labels=[1, -1])
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0
            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if cm[1, 0] + cm[1, 1] > 0 else 0

            if sensitivity > best_sensitivity:
                best_sensitivity = sensitivity
                best_n_components = n_components
                best_accuracy = current_accuracy
                best_specificity = specificity

        self.n_components = best_n_components
        self.pca = PCA(n_components=best_n_components)
        self.fit(X_train_pure, y_train_pure)
        self.threshold_T2, self.threshold_Q

        if self.plotar_DDSIMCA:
            true_labels = np.where(y_test == self.inlier_class, 1, -1)
            predictions, T2, Q = self.predict(X_test)
            cm = confusion_matrix(true_labels, predictions)

            original_labels = ["Adulterated", "Pure"]

            plt.figure(figsize=(8, 6))
            heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                  xticklabels=original_labels, yticklabels=original_labels,
                                  annot_kws={"size": 18})

            colorbar = heatmap.collections[0].colorbar
            colorbar.ax.tick_params(labelsize=14)

            plt.title(f'Best Accuracy={best_accuracy:.2f}', fontsize=16)
            plt.xlabel('Predicted', fontsize=15)
            plt.ylabel('True', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            plt.savefig(f"{self.file_name_no_ext}_Confusion_Matriz_DDSIMCA.tiff", format='tiff')
            plt.show()
            plt.close()

            predictions, T2, Q = self.predict(X_test)
            self.plot_acceptance(T2, Q, y_test, predictions)


            # Exibir amostras incorretamente classificadas
            incorrect_samples = []
            for idx, (true_label, pred_label) in enumerate(zip(true_labels, predictions)):
                if true_label != pred_label:
                    incorrect_sample_info = {
                        'Index': sub_Ys[non_pure_mask].index[idx],  # Índice da amostra no conjunto original de dados
                        'True Label': 'Pure' if true_label == 1 else 'Non-Pure',
                        'Predicted Label': 'Pure' if pred_label == 1 else 'Non-Pure',
                        'Column 2 Value': sub_Ys[non_pure_mask].iloc[idx, 2]  # Valor da coluna 2
                    }
                    incorrect_samples.append(incorrect_sample_info)

            if incorrect_samples:
                print("Amostras classificadas incorretamente:")
                for sample in incorrect_samples:
                    print(f"Índice: {sample['Index']}, Rótulo Verdadeiro: {sample['True Label']}, Rótulo Previsto: {sample['Predicted Label']}, Valor da Coluna 2: {sample['Column 2 Value']}")

                column_2_values = [sample['Column 2 Value'] for sample in incorrect_samples]
                plt.figure(figsize=(10, 6))

                # Ajustar o número de bins conforme necessário, mas aqui usaremos 10 como base
                plt.hist(column_2_values, bins=10, color='skyblue', edgecolor='black')

                plt.title('Histogram of Misclassified Samples', fontsize=16)
                plt.xlabel('Adulteration Percentage (%)', fontsize=15)
                plt.ylabel('Frequency', fontsize=15)

                # Definindo o intervalo do eixo X para 5 em 5
                x_ticks = range(0, int(max(column_2_values)) + 5, 5)  # Ajusta até o máximo valor + 5
                plt.xticks(x_ticks, fontsize=14)
                plt.yticks(fontsize=14)
                plt.ylim(0, 80)  # Define os limites do eixo Y


                # Salvando a figura
                plt.savefig(f"{self.file_name_no_ext}_Histogram_DDSIMCA.tiff", format='tiff')
                plt.show()
                plt.close()  # Fecha a figura após salvar

            else:
                print("Nenhuma amostra foi classificada incorretamente.")

        return best_accuracy, best_n_components, best_sensitivity, best_specificity
    
class OneClassPLS:
    def __init__(self, n_components, inlier_class, n_splits=20, plotar=False, file_name_no_ext="FilePLS", coluna_y_nome="Target"):
        self.n_components = n_components
        self.inlier_class = inlier_class
        self.pls = PLSRegression(n_components=n_components)
        self.scaler = StandardScaler()
        self.threshold = None
        self.n_splits = n_splits
        self.plotar = plotar
        self.file_name_no_ext = file_name_no_ext
        self.coluna_y_nome = coluna_y_nome

    def fit(self, X, y):
        y = np.array([item.strip() for item in y.tolist()])
        X_inliers = X[y == self.inlier_class] # Seleciona apenas os dados que pertencem à classe "inlier"
        X_scaled = self.scaler.fit_transform(X_inliers) # Standardiza os dados
        self.pls.fit(X_scaled, X_scaled) # Ajusta o modelo final usando todos os dados inliers
        X_scores = self.pls.transform(X_scaled) # Projeta os dados no espaço latente
        distances = np.linalg.norm(X_scores, axis=1) # Calcula a distância para o centro no espaço latente
        self.threshold = np.mean(distances) + 2 * np.std(distances) # Calcula o threshold (média + 2 desvios padrão)

    def predict(self, X):
        X_scaled = self.scaler.transform(X) # Padronizando a matriz
        X_scores = self.pls.transform(X_scaled) # Projetando a data no PLS latent space
        distances = np.linalg.norm(X_scores, axis=1) # Computando a distanca da media no latent space
        return np.where(distances <= self.threshold, 1, -1) # Classificando como inlier (1) se T² e Q aestao abaixo do thresholds, ou denomina outlier (-1)

    def fit_and_evaluate_full_pipeline(self, df_pp, sub_Ys, coluna_predicao, plotar):
        # Preparação dos dados
        sub_Ys = sub_Ys.reset_index(drop=True)
        df_pp = df_pp.reset_index(drop=True)

        pure_mask = sub_Ys.loc[:, coluna_predicao].str.replace(" ", "") == self.inlier_class.replace(" ", "")
        non_pure_mask = ~pure_mask

        # Dados inliers
        X_inliers = df_pp[pure_mask].values
        y_inliers = sub_Ys.loc[pure_mask, coluna_predicao].values

        # Dados outliers
        X_outliers = df_pp[non_pure_mask].values
        y_outliers = sub_Ys.loc[non_pure_mask, coluna_predicao].values

        # Dividir inliers em 80% para treino e 20% para teste
        X_train_pure, X_test_pure, y_train_pure, y_test_pure = train_test_split(
            X_inliers, y_inliers, test_size=0.2, random_state=42, shuffle=True
        )

        # Conjunto de teste: 20% dos inliers + todos os outliers
        X_test = np.vstack([X_test_pure, X_outliers])
        y_test = np.hstack([y_test_pure, y_outliers])

        best_sensitivity = 0
        best_n_components = 2
        best_accuracy = 0
        best_specificity = 0

        # Testa diferentes números de componentes e seleciona com base na melhor sensibilidade
        for n_components in range(2, 11):
            self.n_components = n_components
            self.pls = PLSRegression(n_components=n_components)

            # Ajusta o modelo nos dados de treino (inliers) e avalia no conjunto de teste (inliers e outliers)
            self.fit(X_train_pure, y_train_pure)
            predictions = self.predict(X_test)

            # Verificação de métricas
            true_labels = np.where(y_test == self.inlier_class, 1, -1)
            current_accuracy = np.mean(predictions == true_labels)

            # Calculando Sensibilidade e Especificidade
            cm = confusion_matrix(true_labels, predictions, labels=[1, -1])
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0
            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if cm[1, 0] + cm[1, 1] > 0 else 0

            # Guarda o número de componentes que resulta na melhor acuracia
            if current_accuracy > best_accuracy:
                best_sensitivity = sensitivity
                best_n_components = n_components
                best_accuracy = current_accuracy
                best_specificity = specificity

        # Usa o melhor número de componentes encontrado
        self.n_components = best_n_components
        self.pls = PLSRegression(n_components=best_n_components)
        self.fit(X_train_pure, y_train_pure)

        if plotar:
          # Criar matriz de confusão
          true_labels = np.where(y_test == self.inlier_class, 1, -1)
          predictions = self.predict(X_test)
          cm = confusion_matrix(true_labels, predictions)


          # Verifique os tamanhos dos arrays
          print(f"True labels shape: {true_labels.shape}, Predictions shape: {predictions.shape}")

          original_labels = ["Non-Pure", "Pure"]  # Ajustar de acordo com a sua classificação

          # Plotar a matriz de confusão
          plt.figure(figsize=(8, 6))
          heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=original_labels, yticklabels=original_labels,
                                annot_kws={"size": 14})

          # Ajustar os parâmetros da barra de cores
          colorbar = heatmap.collections[0].colorbar
          colorbar.ax.tick_params(labelsize=14)

          # Título e rótulos
          plt.title(f'Best Accuracy={best_accuracy:.2f}', fontsize=16)  # Corrigido para título
          plt.xlabel('Predicted', fontsize=15)
          plt.ylabel('True', fontsize=15)
          plt.xticks(fontsize=14)
          plt.yticks(fontsize=14)

          # Salvar o gráfico da matriz de confusão
          plt.savefig(f'{self.file_name_no_ext}_{self.coluna_y_nome}_Best_Accuracy={best_accuracy:.2f}_OneClassPLS.png', format='png')
          plt.show()
          plt.close()

          # Exibir amostras incorretamente classificadas
          incorrect_samples = []
          for idx, (true_label, pred_label) in enumerate(zip(true_labels, predictions)):
            if true_label != pred_label:
                incorrect_sample_info = {
                    'Index': idx,  # Índice da amostra no conjunto original de dados
                    'True Label': 'Pure' if true_label == 1 else 'Non-Pure',
                    'Predicted Label': 'Pure' if pred_label == 1 else 'Non-Pure',
                    'Column 2 Value': y_test[idx]  # Valor da coluna 2
                }
                incorrect_samples.append(incorrect_sample_info)

          if incorrect_samples:
              print("Amostras classificadas incorretamente:")
              for sample in incorrect_samples:
                  print(f"Índice: {sample['Index']}, Rótulo Verdadeiro: {sample['True Label']}, Rótulo Previsto: {sample['Predicted Label']}, Valor da Coluna 2: {sample['Column 2 Value']}")

              column_2_values = [sample['Column 2 Value'] for sample in incorrect_samples]
              plt.figure(figsize=(10, 6))
              plt.hist(column_2_values, bins=10, color='skyblue', edgecolor='black')
              plt.title('Histograma dos Valores da Coluna 2 (Amostras Classificadas Incorretamente)', fontsize=16)
              plt.xlabel('Valor da Coluna 2', fontsize=14)
              plt.ylabel('Frequência', fontsize=14)
              plt.xticks(fontsize=12)
              plt.yticks(fontsize=12)
              plt.show()
              plt.close()
          else:
              print("Nenhuma amostra foi classificada incorretamente.")
        return best_accuracy, best_n_components, best_sensitivity, best_specificity