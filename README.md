This Readme should help you navigate through the files to reproduce the experiments of this paper.


NOTE: ALL CONFIG FILES SHOULD BE IN THE SAME FOLDER AS THE PYTHON FILES. MOVE THE FILES OR ADJUST THE PATH ACCORDINGLY.

There are three types of experiments conducted in this work:

	1. Effect of pruning on conditioning and training. For this, please refer to the folder Pruning experiments and read the README therein.
	
	2. Condition number at initialization for different linear network architectures. The corresponding config file is config_init_experiments.yaml and the python file initialization_experiments.py
		2.2. For CNNs at initialization, please use config_init_experiments_CNN.yaml and and the python file initialization_experiments_CNN.py instead.
		
	3. Evaluating the condition number during training. For this you need the two config files: config.yaml to specify the training setting. Checkpoints of the models are made and saved in the folder trained_models. 
	These are then reloaded with trained_experiments.py and the corresponding config file config_trained_experiments.yaml
	
	
	In all cases, a pandas Dataframes is created and saved in the folder "pandas_dataframes". They are then used in the jupyter notebook make_seaborn_plots.ipynb
	
Regarding the datasets. You can find cifar-10 in a reformatted manner under the following link: https://file.io/al8yuFzAI4Xr
