# %%
import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder,TargetEncoder
from sklearn.compose import ColumnTransformer
from lib.preprocess import Preprocess
import optuna
from sklearn.model_selection import StratifiedKFold
from functools import partial
from sklearn.utils import compute_sample_weight
import argparse
# %%
arg = argparse.ArgumentParser()
arg.add_argument("--n_trials", type=int, default=0)
n_trials = arg.parse_args().n_trials
# %%
def kfold_cv(params:dict,x_train:pd.DataFrame,y_train:pd.Series,preprocess:ColumnTransformer):
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    #この関数内ではuserwarningを非表示にする
    import warnings
    warnings.simplefilter('ignore')

    #損失を記録するリスト
    losses = []
    #各foldで学習
    for train_index,test_index in skf.split(x_train,y_train):
        #train,testのデータを作成
        X_train_fold,X_test_fold = x_train.iloc[train_index],x_train.iloc[test_index]
        Y_train_fold,Y_test_fold = y_train.iloc[train_index],y_train.iloc[test_index]

        #前処理
        X_train_fold = preprocess.fit_transform(X_train_fold, Y_train_fold)
        X_test_fold = preprocess.transform(X_test_fold)

        X_train_fold = pd.DataFrame(X_train_fold,columns=preprocess.get_feature_names_out())
        X_test_fold = pd.DataFrame(X_test_fold,columns=preprocess.get_feature_names_out())


        cat_cols = X_train_fold.filter(like='cat__').columns.tolist()
        #カテゴリー変数をカテゴリー型のデータに変換
        for col in cat_cols:
            X_train_fold[col] = X_train_fold[col].astype('category')
            X_test_fold[col] = pd.Categorical(X_test_fold[col],categories=X_train_fold[col].cat.categories)


        #dataset,add competition weight using sklearn
        trainset = lgb.Dataset(X_train_fold,label=Y_train_fold)
        # trainset = lgb.Dataset(X_train_fold,label=Y_train_fold)
        testset = lgb.Dataset(X_test_fold,label=Y_test_fold,reference=trainset)

        #モデルを作成
        model = lgb.train(
            params,
            trainset,
            num_boost_round=10000,
            valid_sets=[trainset,testset],
            valid_names=['train','test'],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=100,
                    verbose=False,
                    )
                    ],
            )
        #testデータで予測
        preds = model.predict(X_test_fold,num_iteration=model.best_iteration)
        preds = pd.DataFrame(preds,columns=["regression"])
        round_f = lambda x: int(np.round(x))
        format_f = lambda x: min(max(x, 1), 3)
        preds = preds["regression"].apply(round_f).apply(format_f)
        #f1スコアを計算
        f1 = f1_score(Y_test_fold,preds,average='macro')
        #f1スコアを記録
        losses.append(f1)
    #f1スコアの平均を返す
    return np.mean(losses)
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
# %%
def mean_f1score(preds:np.ndarray,eval_data: lgb.Dataset):
    y_true = eval_data.get_label()
    weight = eval_data.get_weight()
    preds = preds.reshape(len(np.unique(y_true)), -1)
    preds = preds.argmax(axis = 0)
    f1 = f1_score(y_true,preds,average='macro',sample_weight=weight)
    return 'f1',f1,True
# %%
def objective(x,y,preprocess,trial:optuna.trial):
    # ハイパーパラメータの探索範囲
    params = {
        'objective': 'mape',
        'metric': 'mape',
        'seed': 42,
        # search space from https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_integration.py
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    # 交差検証のF1スコアの平均値を返す
    return kfold_cv(params,x,y,preprocess)
ignore_columns = [
    "nta_name",
    "boro_ct",
    "spc_latin",
]
# get object columns
object_columns = [col for col in x_train.select_dtypes(include=["object"]).columns.tolist() if col not in ignore_columns]
config = {
    "object_columns": object_columns,
}
preprocess = Preprocess(config)
# %%
if n_trials:
    objective_fixed = partial(objective,x_train,y_train,preprocess)
    study = optuna.create_study(direction='maximize')
    study.optimize(
        objective_fixed,
        n_trials=n_trials,
        )
# %%
train_index = x_train.index
valid_index = x_valid.index
test_index = x_test.index

x_train = preprocess.fit_transform(x_train, y_train)
x_valid = preprocess.transform(x_valid)
x_test = preprocess.transform(x_test)

y_map_dict = {
    0: 0,
    1: 1,
    2: 2,
}
y_inmap_dict = {v:k for k, v in y_map_dict.items()}

rand_map = lambda x: x + np.random.rand()# * 0.1
y_train = y_train.map(y_map_dict)#.map(rand_map)
y_valid = y_valid.map(y_map_dict)

lgb_train = lgb.Dataset(x_train, y_train)
# lgb_train = lgb.Dataset(x_train, y_train)

lgb_valid = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

params = {
    'objective': 'mape',
    'metric': 'mape',
    'seed': 42,
    'num_threads': -1,
}
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=['train', 'valid'],
    num_boost_round=1000,
    # early_stopping_rounds=100,
    # verbose_eval=100,
    # feval=mean_f1score,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        # lgb.reset_parameter(learning_rate=lambda current_round: 0.01 * 0.995 ** current_round),
    ]
)
# %%
pred_valid = pd.DataFrame(model.predict(x_valid), columns=["health"], index=valid_index)
round_f = lambda x: int(np.round(x))
format_f = lambda x: min(max(x, 1), 3)
pred_valid = pred_valid["health"].apply(round_f).apply(format_f)
print(f1_score(y_valid, pred_valid, average="macro"))
# %%
print(pred_valid.value_counts()/len(pred_valid))

# %%
submit_name = "optuna_lgbm_regression"
predict = pd.DataFrame(model.predict(x_test), columns=["health"], index=test_index)

predict = predict.loc[:, "health"].apply(round_f).apply(format_f).map(y_inmap_dict)
predict.to_csv(f"submission/{submit_name}_submission.csv",  header=False)

# %%
import pickle
with open(f"model/{submit_name}_model.pkl", "wb") as f:
    pickle.dump(model, f)
if n_trials:
    with open(f"model/{submit_name}_study.pkl", "wb") as f:
        pickle.dump(study, f)