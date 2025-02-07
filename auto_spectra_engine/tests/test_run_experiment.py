import os
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock

# Import the functions from your module
from auto_spectra_engine.experiment import run_all_experiments

class TestExperiment(unittest.TestCase):

    @patch('auto_spectra_engine.experiment.load_data')
    @patch('auto_spectra_engine.experiment.insert_results_subpath')
    @patch('auto_spectra_engine.experiment.plot_performance_comparison')
    @patch('auto_spectra_engine.experiment.split_spectrum_from_csv')
    @patch('auto_spectra_engine.experiment.append_results_to_csv')
    @patch('auto_spectra_engine.experiment.mean_centering')
    @patch('auto_spectra_engine.experiment.autoscaling')
    @patch('auto_spectra_engine.experiment.smoothing')
    @patch('auto_spectra_engine.experiment.first_derivative')
    @patch('auto_spectra_engine.experiment.second_derivative')
    @patch('auto_spectra_engine.experiment.msc')
    @patch('auto_spectra_engine.experiment.snv')
    @patch('auto_spectra_engine.experiment.iso_forest_outlier_removal')
    @patch('auto_spectra_engine.experiment.plot_PCA')
    @patch('auto_spectra_engine.experiment.get_plsr_performance')
    @patch('auto_spectra_engine.experiment.get_plsda_performance')
    @patch('auto_spectra_engine.experiment.get_RF_performance')
    @patch('auto_spectra_engine.experiment.OneClassPLS')
    @patch('auto_spectra_engine.experiment.DDSIMCA')
    def test_run_all_experiments(self, mock_DDSIMCA, mock_OneClassPLS, mock_get_RF_performance, mock_get_plsda_performance, mock_get_plsr_performance, mock_plot_PCA, mock_iso_forest_outlier_removal, mock_snv, mock_msc, mock_second_derivative, mock_first_derivative, mock_smoothing, mock_autoscaling, mock_mean_centering, mock_append_results_to_csv, mock_split_spectrum_from_csv, mock_plot_performance_comparison, mock_insert_results_subpath, mock_load_data):
        # Mock return values
        mock_load_data.return_value = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0]
        })
        mock_get_plsr_performance.return_value = (5, 0.1, 0.2, 0.3, 1.2, 2.3, 0.9, 0.8, 0.7)
        mock_get_plsda_performance.return_value = (5, 0.85)
        mock_get_RF_performance.return_value = (0.85, 0.05, 0.8, 0.9, 42, 0.82, 0.06, 0.78, 0.86, 0.84, 0.07, 0.8, 0.88)
        mock_OneClassPLS.fit_and_evaluate_full_pipeline.return_value = (0.8, 5, 0.7, 0.75)
        mock_DDSIMCA.fit_and_evaluate_full_pipeline.return_value = (0.88, 5, 0.85, 0.9)
        mock_insert_results_subpath.return_value = "results/test_output"
        mock_iso_forest_outlier_removal.return_value = (pd.DataFrame({"feature1": [1.0], "feature2": [4.0]}), pd.DataFrame({"target": [0]}))

        # Call the function
        file = "raman.csv"
        result = run_all_experiments(file, modelo="PLSDA", coluna_predicao="Adulterant", pipeline_family="Raman", use_split_spectra=4)

        # Assertions
        self.assertIsNotNone(result)
        mock_load_data.assert_called_once()
        mock_get_plsda_performance.assert_called_once()
        mock_append_results_to_csv.assert_called()

if __name__ == '__main__':
    unittest.main()