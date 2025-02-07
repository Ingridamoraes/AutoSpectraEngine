import auto_spectra_engine.experiment as ase

file = "/home/barbon/Python/AutoSpectraEngine/auto_spectra_engine/datasets/raman.csv"
ase.run_all_experiments(file, modelo="PLSDA", coluna_predicao="Adulterant", pipeline_family="Raman", use_split_spectra=10)
