 Data Drive Security Insights

Data Drive Security Insights is a machine learning project focused on analyzing and predicting the nature and impact of cybersecurity breaches. Using breach data, this project classifies breach methods and predicts the estimated impact (e.g., number of records lost), helping inform proactive data protection strategies.

Project Objectives

- Classify: The breach method (e.g., Hacking, Phishing, Lost Device).
- Predict: Breach impact using regression models (e.g., records affected).
- Visualize: breach trends, methods, and impacts.
- Support:  data-driven security insights for decision-makers.

Project Structure:
├── data/             #Dataset 

├── notebooks/        #Jupyter notebooks for EDA, ML, visualization

├── src/              #Modular Python scripts for data & model logic

├── reports/          #Visualizations and analysis results

├── requirements.txt  #Python dependencies

├── README.md         #Project overview 

├── LICENSE 


*Dependencies:
Install the required Python packages:

Bash: 
pip install -r requirements.txt

Main Packages Used:
-pandas
-scikit-learn
-matplotlib
-seaborn
-jupyter

How to Use: 
git clone https://github.com/GoatVenom/Data-Driven-Security-Insights.git
cd Data-Drive-Security-Insights


Launch Jupyter Notebook:
jupyter notebook

Open notebooks in order:
01_data_preprocessing.ipynb
02_breach_method_classifier.ipynb
03_impact_prediction.ipynb
04_data_visualization.ipynb

Modify or reuse functions from the src/ folder if working in scripts.

Visualizations include:
Breach imapact frequency by year
Confusion matrix for method classifier
Impact prediction scatterplot

All saved in reports/figures.

Project Usage:
This project can be used to:
* Monitor breach trends
* Simulate impact forecasting for hypothetical breaches
* Explore which factors most influence breach consequences
