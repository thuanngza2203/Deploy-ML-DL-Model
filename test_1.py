import pickle 

my_model_clf = pickle.load(open("model.pkl", 'rb'))

# file.json
import json
import pandas as pd

data = pd.read_json('file (1).json')
data = pd.DataFrame(data)
print(data)

predictions = my_model_clf.predict(data)
print(predictions)