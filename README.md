Optimization of SCC with Recycled Coarse Aggregate and Quarry Dust as Partial Replacement Using Machine Learning Techniques (Hybrid Models)
A hybrid Artificial Neural Network (ANN) + Genetic-Algorithm (GA) + Support Vector Regressor (SVR) stacking pipeline for predicting the compressive strength of Self Compacting Concrete (SCC) mixes made with two partial-replacement materials:
Quarry Dust (QD) — a partial replacement for fine aggregate
Recycled Coarse Aggregate (RCA) — a partial replacement for coarse aggregate

____________________________________________________________________________________________________________________________________________________________________________________________


Table of Contents
Description
Datasets
Key Finding
Built With
Installation
How to run the project
Outputs
Learning Outcomes
Project Structure
License

____________________________________________________________________________________________________________________________________________________________________________________________


Description
Self-Compacting Concrete (SCC) is prized for its excellent flowability and compaction without vibration, but its raw-material footprint (cement, natural fine/coarse aggregate) is costly and carbon-intensive. Replacing part of the fine aggregate with quarry dust (a stone-crushing by-product) and part of the coarse aggregate with recycled concrete aggregate reduces cost and environmental impact — but makes strength prediction harder, since replacement content interacts non-linearly with mix proportions, supplementary cementitious materials (fly ash, GGBS, silica fume), and curing age.
 
An Artificial Neural Network (ANN) learns the overall non-linear
mix–strength relationship.
A Support Vector Regressor (SVR, RBF kernel) is tuned by a
Genetic Algorithm that searches to maximize
validation-set R².

____________________________________________________________________________________________________________________________________________________________________________________________


Datasets
Dataset	File	Samples	Replacement material
Quarry Dust	data/concrete_QD_data.csv	278	Fine aggregate → Quarry Dust
Recycled Coarse Aggregate	data/concrete_RCA_data.csv	160	Coarse aggregate → RCA

____________________________________________________________________________________________________________________________________________________________________________________________

Key Finding

Quarry Dust (QD) — 278 samples
Split	RMSE	MAE	R²	MAPE
Train	2.09	1.61	0.983	5.39%
Validation	3.89	2.77	0.947	8.59%
Test	3.91	2.55	0.938	6.06%

Recycled Coarse Aggregate (RCA) — 160 samples
Split	RMSE	MAE	R²	MAPE
Train	4.05	3.19	0.894	11.14%
Validation	6.79	4.09	0.737	19.70%
Test	5.39	4.42	0.792	16.08%

____________________________________________________________________________________________________________________________________________________________________________________________


Built With
Python 3.11
Pandas — Data processing
Numpy — Numerical computing
Scikit-Learn — Machine learning utilities
joblib — Nonlinear time-series forecasting
Matplotlib — Visulization
Jupyter — notebooks

____________________________________________________________________________________________________________________________________________________________________________________________

 
Installation
Step 1: Install Anaconda

Step 2: Download the project
	Download the project from GitHub or clone the repository.

Step 3: Open Anaconda Prompt

Step 4: Navigate to the Project folder
	---bash
	cd "File location"

Step 5: Create the Conda Environment
	---bash
	conda env create -f environment.yml
	"This command automatically install all required packages.

Step 6: Activate the Environment
	---bash
	conda activate scc-ml-hybrid
	pip install -r requirements.txt

Full analysis on Traffic data:
Download data from GitHub.
Place the csv in (data\)
   or you can load and filter to your locations of interest.

____________________________________________________________________________________________________________________________________________________________________________________________


How to run the Project
Step 1: Open Anaconda prompt

Step 2: Navigate to the Project folder
	---bash
	cd "File location"

Step 3: Run the file
	---bash
	python run.py --dataset both --quick

Step 4: open outputs

____________________________________________________________________________________________________________________________________________________________________________________________


Outputs
Optimized SCC Mix Design
Machine Learning Prediction Model
Model Performance Evaluation
Optimization Results
Graphs and Visualizations
___________________________________________________________________________________________________________________________________________________________________________________________


Learning Outcomes
Understand the properties of Self-Compacting Concrete (SCC).
Build machine learning models for predicting concrete properties.
Visualize and communicate machine learning results using graphs and tables.
Gain practical experience in applying AI and machine learning to civil engineering materials.

____________________________________________________________________________________________________________________________________________________________________________________________


Project Structure

scc_ml_hybrid/
├── README.md
├── environment.yml
├── requirements.txt
├── data/
│   ├── concrete_QD_data.csv
│   └── concrete_RCA_data.csv
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── genetic_algorithm.py
│   ├── models.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train_pipeline.py
│   └── visualize.py
└── outputs/
    │── metrics/
    │	├── dataset_comparison.csv
    │	├── ga_history_qd.csv
    │	├── ga_history_rca.csv
    │ 	├── metrics_qd.csv
    │	├── metrics_rca.csv
    │	├── predictions_qd.csv
    │	└── predictions_rca.csv
    └── plots/
	├── actual_vs_predicted_QD.png
        ├── actual_vs_predicted_RCA.png
   	├── ann_convergence_QD.png
   	├── ann_convergence_RCA.png
   	├── dataset_comparison.png
   	├── error_percent_QD.png
   	├── error_percent_RCA.png
   	├── ga_convergence_QD.png
   	└── ga_convergence_RCA.png

___________________________________________________________________________________________________________________________________________________________________________________________

License
This project is released under the MIT License.
You are free to use, modify, and distribute this work with proper attribution.
See the `LICENSE` file for complete license information.
____________________________________________________________________________________________________________________________________________________________________________________________
