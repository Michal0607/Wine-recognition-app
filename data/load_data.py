from datasets import load_dataset
import pandas as pd

dataset = load_dataset("katossky/wine-recognition")
data = pd.DataFrame(dataset['train'])

data.to_csv('/data/wine_data.csv', index=False)
