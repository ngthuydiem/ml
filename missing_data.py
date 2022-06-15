# setup libraries

import pandas as pd
import numpy as np
from typing import NamedTuple

import matplotlib.pylab as plt
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']

import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')


class Data(NamedTuple):
	train: pd.DataFrame
	test: pd.DataFrame
	ss: pd.DataFrame # sample submission
	tt: pd.DataFrame # combined train test


def read_kaggle_data(input_path="../input"):
	train = pd.read_csv(f"{input_path}/train.csv")
	test = pd.read_csv(f"{input_path}/test.csv")
	ss = pd.read_csv(f"{input_path}/sample_submission.csv")

	train["isTrain"] = True
	test["isTrain"] = False

	tt = pd.concat([train, test]).reset_index(drop=True).copy()

	print(f"Train shape: {train.shape}. Test shape: {test.shape}. Train test shape: {tt.shape}")
	return Data(train, test, ss, tt)


def show_missing_values(data):
	ncounts = pd.DataFrame([data.train.isna().mean(), data.test.isna().mean()]).T
	ncounts = ncounts.rename(columns={0: "train_missing", 1: "test_missing"})

	ncounts_train_missing = ncounts.query("train_missing > 0") 
	ncounts_train_missing.plot(
	    kind="barh", figsize=(8, 5), title="% of Values Missing"
	)
	plt.show()

	return list(ncounts_train_missing.index) # names of fields with missing values


def find_missing_values_per_observation(data, nacols):
	data.tt["n_missing"] = data.tt[nacols].isna().sum(axis=1)
	data.train["n_missing"] = data.train[nacols].isna().sum(axis=1)
	data.test["n_missing"] = data.test[nacols].isna().sum(axis=1)

	data.tt["n_missing"].value_counts().plot(
	    kind="bar", title="Number of Missing Values per Sample"
	)
	plt.show()


def evaluate_missing_indicators(data, nacols, target):

	# Try to predict the target using only missing value indicators as features

	from sklearn.linear_model import LogisticRegressionCV
	from sklearn.metrics import roc_auc_score

	tt_missing_tag_df = data.tt[nacols].isna()
	tt_missing_tag_df.columns = [f"{c}_missing" for c in tt_missing_tag_df.columns]
	tt = pd.concat([data.tt, tt_missing_tag_df], axis=1)

	missing_indicators = list(tt_missing_tag_df.columns)
	lr = LogisticRegressionCV(scoring="accuracy")
	X = tt.query("isTrain")[missing_indicators]
	y = tt.query("isTrain")[target]

	lr.fit(X, y)
	lr.score(X, y)

	preds = lr.predict_proba(X)[:, 0]

	return roc_auc_score(y, preds)