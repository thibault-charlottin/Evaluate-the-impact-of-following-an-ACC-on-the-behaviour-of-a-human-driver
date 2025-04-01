# Evaluate the impact of following an ADAS on the behaviour of a human driver

This repository contains all the code that you will need to reproduce the figures and models that were used to write the paper "Does following an ADAS change the behaviour of a driver?" <br>.
The repository is organised as follows:
```
📦Evaluate the impact of following an ADAS on the behaviour of a human driver
 ┣ 📂data
 ┃ ┣ 📂raw_data
 ┃ ┗ 📂by_run
 ┣ 📂out
 ┃ ┣ 📂data
 ┃ ┃ ┣ 📂DTW
 ┃ ┃ ┣ 📂LC
 ┃ ┃ ┗ 📂tests
 ┃ ┗ 📂images
 ┣ 📂demo
 ┃ ┗ 📜demo.ipynb
 ┣ 📂env
 ┃ ┗ 📜ADAS_HDV_interraction.yml
 ┣ 📂src
 ┃ ┣ 📜add_safety_indicators.py
 ┃ ┣ 📜add_stimulus_evaluation.py
 ┃ ┣ 📜cftest.py
 ┃ ┣ 📜compare_cf_behaviour.py
 ┃ ┣ 📜compare_cf_execution.py
 ┃ ┣ 📜compare_safety.py
 ┃ ┣ 📜compare_stimulus_reaction.py
 ┃ ┣ 📜detect_LC.py
 ┃ ┣ 📜model_LC_proba.py
 ┃ ┣ 📜prepare_data_for_LC_logit.py
 ┃ ┗ 📜read_data.py
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┗📜console.ipynb
```

To run the code, use the notebook "console.ipynb". To run an example of the preprocessing, run "demo/demo.ipynb"

To install the necessary packages, follow the guidelines below. Be aware that they differ depending on whether you are a Windows user or a Unix kernel-based OS user.

### Unix distributions/MacOS installation

Copy your local path to this repository.
Then open the command prompt.

````bash
cd %paste your path
````

````bash
conda env create -f env/ADAS_HDV_interraction.yml
````

Activate it:
````bash
conda activate ADAS_HDV_interraction
````

You can then run the commands in the console.ipynb file 

### Windows installation
Copy your local path to this repository.
Open Anaconda navigator.
Open CMD.exe prompt and type
````bash
cd %paste your path
````

then type 
````bash
conda env create -f env/ADAS_HDV_interraction.yml
````

Activate it:
````bash
conda activate ADAS_HDV_interraction
````

You can then run the commands in the console.ipynb file 

## download data
To download the TGSIM dataset, follow the links in the data/raw_data folder. Preprocessed data is available on request.<br>
Preprocessed data is to be added into a folder named "by_run" daughter of the mother folder "data".

### happy coding!
