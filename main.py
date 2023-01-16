import numpy as np
import pandas as pd
from easydict import EasyDict
import random
import scipy
from tqdm import tqdm

configuration = EasyDict({
    "input_dir":'/kaggle/working/',
    "seed":40,
    "n_folds":5,
    "target":'target',
    "boosting_type":'dart',
    "metric":'binary_logloss',
    "cat_features":[
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
})

random.seed(configuration.seed)
np.random.seed(configuration.seed)


def amex_metric(y_real, y_pred):
    # arraying all the lables
    data_labels = np.array([y_real, y_pred]).T
    #  lets arrange lables in decending order on y_pred
    data_labels = data_labels[np.argsort(-y_pred)]
    wei = np.where(data_labels[:, 0] == 0, 20, 1)
    # Get top lables
    c_vals = data_labels[np.cumsum(wei) <= int(0.04 * np.sum(wei))]
    t_four = np.sum(c_vals[:, 0]) / np.sum(data_labels[:, 0])
    #  gini array
    gini = [0, 0]
    for i in [1, 0]:
        data_labels = np.array([y_real, y_pred]).T
        data_labels = data_labels[np.argsort(-y_pred if i else -y_real)]
        # Assign weights
        weight = np.where(data_labels[:, 0] == 0, 20, 1)
        # weight_random calculation
        random_weight = np.cumsum(weight / np.sum(weight))
        t_pos = np.sum(data_labels[:, 0] * weight)
        cu_p_found = np.cumsum(data_labels[:, 0] * weight)
        lorentz = cu_p_found / t_pos
        # Calculate gini
        gini[i] = np.sum((lorentz - random_weight) * weight)
    # Return final metric
    return 0.5 * (gini[1] / gini[0] + t_four)



def amex_metric_lightgbm(y_pred, y_real):
    # obtain true labels
    y_real = y_real.get_label()
    # Calculate amex_metric
    m_value = amex_metric(y_real, y_pred)
    return 'amex_metric', m_value, True


#  code for data Preprocessing
def extract_difference(data, num_features):
    """
    Function to calculate the difference of numeric features in a dataframe and group by 'customer_ID'
    """
    # Initialize lists
    data_f1 = []
    cus_ids = []

    # Iterate over groups of dataframe,
    for cus_id, df in tqdm(data.groupby(["customer_ID"])):  # grouped by 'customer_ID'
        # Calculate the differences of num_features
        d_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        data_f1.append(d_df1)
        cus_ids.append(cus_id)
    # join the differences to the  ids of customers

    data_f1 = np.concatenate(data_f1, axis=0)
    # Transform to dataframe
    data_f1 = pd.DataFrame(data_f1, columns=[col + "_diff1" for col in df[num_features].columns])
    # Add customer id
    data_f1["customer_ID"] = cus_ids
    return data_f1

def get_preprocess_data():
    """
    Function to datapreprocess
    """
    print("Starting to read the data")
    # get train data
    train_data = pd.read_csv("/kaggle/input/amex-default-prediction/train_data.csv")
    # extract categorical and numerical features
    features = train_data.drop(["customer_ID", "S_2"], axis=1).columns.to_list()
    catogorical_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    n_feat = [col for col in features if col not in catogorical_features]
    # Aggregate numerical features
    tr_numerical_agg = train_data.groupby("customer_ID")[n_feat].agg(["mean", "std", "min", "max", "last"])
    tr_numerical_agg.columns = ["_".join(x) for x in tr_numerical_agg.columns]
    tr_numerical_agg.reset_index(inplace=True)
    # Aggregate categorical features
    tr_catogarical_agg = train_data.groupby("customer_ID")[catogorical_features].agg(["count", "last", "nunique"])
    tr_catogarical_agg.columns = ["_".join(x) for x in tr_catogarical_agg.columns]
    tr_catogarical_agg.reset_index(inplace=True)
    # training labels reading
    train_labels = pd.read_csv("/kaggle/input/amex-default-prediction/train_labels.csv")
    cols = list(tr_numerical_agg.dtypes[tr_numerical_agg.dtypes == "float64"].index)
    for col in tqdm(cols):
        tr_numerical_agg[col] = tr_numerical_agg[col].astype(np.float32)
    cols = list(tr_catogarical_agg.dtypes[tr_catogarical_agg.dtypes == "int64"].index)
    for col in tqdm(cols):
        tr_catogarical_agg[col] = tr_catogarical_agg[col].astype(np.int32)
    train_diff = extract_difference(train_data, n_feat)
    # code to Merge the aggregated features, differences, and labels
    train_data = tr_numerical_agg.merge(
        tr_catogarical_agg, how="inner", on="customer_ID"
    ).merge(train_diff, how="inner", on="customer_ID").merge(
        train_labels, how="inner", on="customer_ID"
    )
    # test data reading and doing the same thing done above for test data
    test = pd.read_parquet("/kaggle/input/amex-default-prediction/test_data.csv")
    print("Starting test feature engineering...")
    te_n_agg = test.groupby("customer_ID")[n_feat].agg(["mean", "std", "min", "max", "last"])
    te_n_agg.columns = ["_".join(x) for x in te_n_agg.columns]
    te_n_agg.reset_index(inplace=True)
    te_c_agg = test.groupby("customer_ID")[catogorical_features].agg(["count", "last", "nunique"])
    te_c_agg.columns = ["_".join(x) for x in te_c_agg.columns]
    te_c_agg.reset_index(inplace=True)
    test_diff = extract_difference(test, n_feat)
    test = te_n_agg.merge(te_c_agg, how="inner", on="customer_ID").merge(
        test_diff, how="inner", on="customer_ID"
    )
    # get the final data as the preprocessed data..........
    return train_data, test

train,test = get_preprocess_data()


def train_data_and_evaluate_model(train_data, test_data):
    #  encode categorical features. im using lable encoder to the data
    catogary_cols = configuration.cat_features
    catogary_cols = [f"{col}_last" for col in catogary_cols]
    for col in catogary_cols:
        train_data[col] = train_data[col].astype('category')
        test_data[col] = test_data[col].astype('category')

    f_cols = train_data.select_dtypes(include=['float']).columns
    f_cols = [col for col in f_cols if 'last' in col]
    train_data[f_cols] = train_data[f_cols].round(2)
    test_data[f_cols] = test_data[f_cols].round(2)

    n_cols = [col for col in train_data.columns if 'last' in col]
    n_cols = [col[:-5] for col in n_cols if 'round' not in col]
    for col in n_cols:
        train_data[f'{col}_last_mean_diff'] = train_data[f'{col}_last'] - train_data[f'{col}_mean']
        test_data[f'{col}_last_mean_diff'] = test_data[f'{col}_last'] - test_data[f'{col}_mean']

    f_cols = train_data.select_dtypes(include=['float']).columns
    train_data[f_cols] = train_data[f_cols].astype(np.float16)
    test_data[f_cols] = test_data[f_cols].astype(np.float16)
    # Get feature list
    features = [col for col in train_data.columns if col not in ['customer_ID', configuration.target]]
    # Define model parameters
    params = {
        'objective': 'binary',
        'metric': configuration.metric,
        'boosting': configuration.boosting_type,
        'seed': configuration.seed,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
    }
    test_predictions = np.zeros(len(test_data))
    # initiate a numpy array for  folds predictions
    of_predictions = np.zeros(len(train_data))
    from sklearn.model_selection import StratifiedKFold, train_test_split
    import lightgbm as lightgbm
    kfold = StratifiedKFold(n_splits=configuration.n_folds, shuffle=True, random_state=configuration.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train_data, train_data[configuration.target])):
        print(f'\nTraining fold {fold} with {len(features)} features...')
        x_train, x_val = train_data[features].iloc[trn_ind], train_data[features].iloc[val_ind]
        y_train, y_val = train_data[configuration.target].iloc[trn_ind], train_data[configuration.target].iloc[val_ind]
        lightgbm_train = lightgbm.Dataset(x_train, y_train, categorical_feature=catogary_cols)
        lightgbm_val = lightgbm.Dataset(x_val, y_val, categorical_feature=catogary_cols)
        model = lightgbm.train(params, lightgbm_train, valid_sets=[lightgbm_train, lightgbm_val],
                               valid_names=['train', 'val'], num_boost_round=1000,
                               early_stopping_rounds=50, verbose_eval=50,
                               feval=amex_metric_lightgbm)
        of_predictions[val_ind] = model.predict(x_val)
        test_predictions += model.predict(test_data[features]) / configuration.n_folds
        score = amex_metric(y_val, model.predict(x_val))
    score = amex_metric(train_data[configuration.target], of_predictions)
    test_df = pd.DataFrame({'customer_ID': test_data['customer_ID'], 'prediction': test_predictions})
    test_df.to_csv(f'/kaggle/working/submission.csv', index=False)

train_data_and_evaluate_model(train, test)