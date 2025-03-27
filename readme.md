# Evaluate the impact of following an ACC on the behaviour of a human driver

This repository contains all the code that you will need to reproduce the figures and models that were used to write the paper "Does following an ADAS change the behaviour of a driver?" <br>.
The repository is organised as follwos:
📦Evaluate the impact of followign an ACC on the behaviour of a human driver
 ┣ env
 ┃ ┗ environment_without_biogeme.yml
 ┣ 📂data
 ┃ ┣ 📂raw_data
 ┃ ┣ 📂by_run
 ┃ ┣ 📂raw_data
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
 ┣ 📜console.ipynb
 ┗ 📜setup.py


To install the necessary packages follow the following guidelines, be aware that they differ whether you are a Windows user or a Unix kernel-based OS user.

### Unix distributions/MacOS installation

Copy your local path to this repository
Then open the command prompt
````bash
cd %paste your path
````

````bash
conda env create -f env/environment_without_biogeme.yml
````

Activate it:
````bash
conda activate ADAS_HDV_interraction
````

You can then run the commands in the console.ipynb file 

### Windows installation
Copy your local path to this repository
Open Anaconda navigator
Open CMD.exe prompt and type
````bash
cd %paste your path
````

then type 
````bash
conda env create -f env/environment_without_biogeme.yml
````

Activate it:
````bash
conda activate ADAS_HDV_interraction
````

You can then run the commands in the console.ipynb file 

