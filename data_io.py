import pandas as pd
from typing import NamedTuple

class Data(NamedTuple):
	train: pd.DataFrame
	test: pd.DataFrame
	ss: pd.DataFrame # sample submission
	tt: pd.DataFrame # combined train test
	features: list
	target: str
	input_path: str


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
	return Data(train, test, ss, tt, features, target, input_path)
