# setup libraries

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']

import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')


def read_kaggle_data(input_path="../input"):
	train = pd.read_csv(f"{input_path}/train.csv")
	test = pd.read_csv(f"{input_path}/test.csv")
	ss = pd.read_csv(f"{input_path}/sample_submission.csv")

	train["isTrain"] = True
	test["isTrain"] = False

	tt = pd.concat([train, test]).reset_index(drop=True).copy()

	print(f"Train shape: {train.shape}. Test shape: {test.shape}. Train test shape: {tt.shape}")
	return train, test, ss, tt


def show_missing_values(train, test):
	ncounts = pd.DataFrame([train.isna().mean(), test.isna().mean()]).T
	ncounts = ncounts.rename(columns={0: "train_missing", 1: "test_missing"})

	ncounts.query("train_missing > 0").plot(
	    kind="barh", figsize=(8, 5), title="% of Values Missing"
	)
	plt.show()
