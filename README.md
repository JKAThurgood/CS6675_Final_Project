# Decision Tree Programming
## Data
Create a directory are route labeled `data`. Download from the following links and extract csvs to the data dir. 
### Airlines
https://www.kaggle.com/datasets/sherrytp/airline-delay-analysis?resource=download  
Download the 2010 instance and extract into a subdirectory `data/airlines` to follow project example. If desired download other years and change the import to see performance across the years. 
### Electricity
https://www.kaggle.com/datasets/datasetengineer/electricity-market-dataset?select=electricity_market_dataset.csv  
Download and extract to `data`
### Weather
https://www.kaggle.com/datasets/apurboshahidshawon/weatherdatabangladesh  
Download and extract to `data`

## Environment
I had to create a python env to run everything. I used the following commands 
```
python3 -m venv decision_trees_env; 
source decision_trees_env/bin/activate; 
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel;
```

## Notebooks
Each dataset gets it's own notebook to run experiments inside of. Feel free to experiment with different windows and retraining RMSE values. Each training method has it's own function file so that updates are easily passed to each of the experiments