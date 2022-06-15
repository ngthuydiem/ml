def setup_notebook():
	import pandas as pd
	import numpy as np
	import matplotlib.pylab as plt
	color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']

	import warnings
	warnings.filterwarnings("ignore")
	plt.style.use('ggplot')


def read_kaggle_data():
	train = pd.read_csv("../input/train.csv")
	test = pd.read_csv("../input/test.csv")
	ss = pd.read_csv("../input/sample_submission.csv")

	train["isTrain"] = True
	test["isTrain"] = False

	tt = pd.concat([train, test]).reset_index(drop=True).copy()
