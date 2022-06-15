# Credit: https://www.kaggle.com/code/robikscube/handling-with-missing-data-youtube-stream/notebook

import pandas as pd
import numpy as np
from typing import NamedTuple
from measure import timeit

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
	features: list
	target: str


def read_kaggle_data(input_path="../input"):
	train = pd.read_csv(f"{input_path}/train.csv")
	test = pd.read_csv(f"{input_path}/test.csv")
	ss = pd.read_csv(f"{input_path}/sample_submission.csv")

	features = list(test.columns)
	target = [x for x in train.columns if x not in features][0]
	
	train["isTrain"] = True
	test["isTrain"] = False

	tt = pd.concat([train, test]).reset_index(drop=True).copy()

	print(f"Train shape: {train.shape}. Test shape: {test.shape}. Train test shape: {tt.shape}")
	return Data(train, test, ss, tt, features, target)


def show_missing_values(data):
	ncounts = pd.DataFrame([data.train.isna().mean(), data.test.isna().mean()]).T
	ncounts = ncounts.rename(columns={0: "train_missing", 1: "test_missing"})

	ncounts_train_missing = ncounts.query("train_missing > 0") 
	try:
		ncounts_train_missing.plot(
		    kind="barh", figsize=(8, 5), title="% of Values Missing"
		)
		plt.show()
	except IndexError:
		print('No missing values!')


	return list(ncounts_train_missing.index) # names of fields with missing values


def find_missing_values_per_observation(data, nacols):
	data.tt["n_missing"] = data.tt[nacols].isna().sum(axis=1)
	data.train["n_missing"] = data.train[nacols].isna().sum(axis=1)
	data.test["n_missing"] = data.test[nacols].isna().sum(axis=1)

	data.tt["n_missing"].value_counts().plot(
	    kind="bar", title="Number of Missing Values per Sample"
	)
	plt.show()


def evaluate_missing_indicators(data, nacols):

	# Try to predict the target using only missing value indicators as features

	from sklearn.linear_model import LogisticRegressionCV
	from sklearn.metrics import roc_auc_score

	tt_missing_tag_df = data.tt[nacols].isna()
	tt_missing_tag_df.columns = [f"{c}_missing" for c in tt_missing_tag_df.columns]
	tt = pd.concat([data.tt, tt_missing_tag_df], axis=1)

	missing_indicators = list(tt_missing_tag_df.columns)
	lr = LogisticRegressionCV(scoring="accuracy")
	X = tt.query("isTrain")[missing_indicators]
	y = tt.query("isTrain")[data.target]

	lr.fit(X, y)
	lr.score(X, y)

	preds = lr.predict_proba(X)[:, 0]

	return roc_auc_score(y, preds)


@timeit
def impute_with_sklearn_simple(data, field): # fast
	from sklearn.impute import SimpleImputer
	imptr = SimpleImputer(strategy="mean", add_indicator=False)

	# Fit / Transform on train, transform only on val/test
	train_imp = imptr.fit_transform(data.train[data.features])
	test_imp = imptr.transform(data.test[data.features])

	# For kaggle competition you can kind of cheat by fitting on all data
	tt_impute = imptr.fit_transform(data.tt[data.features])
	tt_simple_impute = pd.DataFrame(tt_impute, columns=data.features)

	tt_simple_impute[field].plot(
		kind='hist',
		bins=50,
		title='Simple Impute',
		color=color_pal[0]
	)

	plt.show()
	return tt_simple_impute


@timeit
def impute_with_sklearn_iterative(data, field): # fast
	from sklearn.experimental import enable_iterative_imputer  # noqa
	from sklearn.impute import IterativeImputer

	it_imputer = IterativeImputer(max_iter=10)
	train_iterimp = it_imputer.fit_transform(data.train[data.features])
	test_iterimp = it_imputer.transform(data.test[data.features])
	tt_iterimp = it_imputer.fit_transform(data.tt[data.features])

	# Create train test imputed dataframe
	tt_iter_imp_df = pd.DataFrame(tt_iterimp, columns=data.features)

	tt_iter_imp_df[field].plot(
		kind='hist',
		bins=50,
		title='Iterative Impute',
		color=color_pal[1]
	)

	plt.show()
	return tt_iter_imp_df



@timeit
def impute_with_knn(data, field): # slowest
	# https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
	from sklearn.impute import KNNImputer

	knn_imptr = KNNImputer(n_neighbors=1)
	train_knnimp = knn_imptr.fit_transform(data.train[data.features])
	test_knnimp = knn_imptr.transform(data.test[data.features])
	tt_knnimp = knn_imptr.fit_transform(data.tt[data.features])
	tt_imp = pd.DataFrame(tt_knnimp, columns=data.features)

	# Create KNN Train/Test imputed dataframe
	knn_imp_df = pd.DataFrame(tt_imp, columns=data.features)

	knn_imp_df[field].plot(
		kind='hist',
		bins=50,
		title='KNN Impute',
		color=color_pal[2]
	)
	plt.show()

	return knn_imp_df

@timeit
def impute_with_lgbm(data, field): # slower
	# https://github.com/analokmaus/kuma_utils/blob/master/preprocessing/imputer.py
	import sys
	sys.path.append("kuma_utils/")
	from kuma_utils.preprocessing.imputer import LGBMImputer

	lgbm_imtr = LGBMImputer(n_iter=100, verbose=True)

	train_lgbmimp = lgbm_imtr.fit_transform(data.train[data.features])
	test_lgbmimp = lgbm_imtr.transform(data.test[data.features])
	tt_lgbmimp = lgbm_imtr.fit_transform(data.tt[data.features])

	tt_imp = pd.DataFrame(tt_lgbmimp, columns=data.features)

	# Create LGBM Train/Test imputed dataframe
	lgbm_imp_df = pd.DataFrame(tt_imp, columns=data.features)

	tt_lgbm_imp = pd.concat([data.tt[["id", "isTrain", data.target]],
	                         tt_lgbmimp], axis=1)

	# check the imputation distribution
	tt_lgbm_imp[field].plot(
		kind='hist',
		bins=50,
		title='LGBM Impute',
		color=color_pal[3]
	)
	plt.show()

	return tt_lgbm_imp