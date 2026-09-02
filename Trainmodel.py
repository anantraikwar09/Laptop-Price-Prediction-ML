import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
df = pd.read_csv("/content/drive/MyDrive/Ai Medicaps(24 08 26)/ensemble ML/Clean_dataset.csv")
x = df.drop(columns = ['Price_inr'])
y = df['Price_inr']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 ,  random_state = 42 )
cat_fea = ['Company','TypeName','Cpu','OpSys','Gpu']
step1 = ColumnTransformer(transformers = [
    ('col_tnf',OneHotEncoder(sparse_output = False, handle_unknown = 'ignore'),cat_fea)
],remainder = 'passthrough')

step2 = GradientBoostingRegressor( n_estimators=500)

pipe = Pipeline([('step1',step1),('step2',step2)])


pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test) 
with open("pipe.pkl", "wb") as f:
    pickle.dump(pipe, f)

with open("df.pkl", "wb") as f:
    pickle.dump(df, f)

print("Saved pipe.pkl and df.pkl")