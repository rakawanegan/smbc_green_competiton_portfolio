# %%
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,TargetEncoder
from sklearn.compose import ColumnTransformer
from lib.preprocess import Preprocess
# %%
SEED = 314
datasrc = "data/official/"
data = pd.read_csv(os.path.join(datasrc, "train.csv"), index_col=0)
x_test = pd.read_csv(os.path.join(datasrc, "test.csv"), index_col=0)

train, valid = train_test_split(data, test_size=0.2, random_state=42, stratify=data["health"])

# %%
x_train = train.drop("health", axis=1)
y_train = train["health"]
x_valid = valid.drop("health", axis=1)
y_valid = valid["health"]

ignore_columns = [
    "nta_name",
    "boro_ct",
    "spc_latin",
]
# get object columns
object_columns = [col for col in x_train.select_dtypes(include=["object"]).columns.tolist() if col not in ignore_columns]
config = {
    "object_columns": object_columns,
    "is_target_encode": False,
}
preprocess = Preprocess(config)
# %%
train_index = x_train.index
valid_index = x_valid.index
test_index = x_test.index

x_train = preprocess.fit_transform(x_train, y_train)
x_valid = preprocess.transform(x_valid)
x_test = preprocess.transform(x_test)

x_train = pd.DataFrame(x_train, index=train_index, columns=preprocess.get_feature_names_out())
x_valid = pd.DataFrame(x_valid, index=valid_index, columns=preprocess.get_feature_names_out())
x_test = pd.DataFrame(x_test, index=test_index, columns=preprocess.get_feature_names_out())
# %%
print("x_train shape: ",x_train.shape)
print("x_valid shape: ",x_valid.shape)
print("x_test shape: ",x_test.shape)
print()
print("x_train columns: ",x_train.info())
# print(x_train.head())
if x_train.columns.tolist() != x_valid.columns.tolist() or x_train.columns.tolist() != x_test.columns.tolist():
    print("columns are not same!")
    print("x_valid columns: ",x_valid.columns)
    print("x_test columns: ",x_test.columns)
else:
    print("columns are same!")
print()
# %%
print("run preprocess.py successfully!")