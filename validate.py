def add_kfold(data):

    from sklearn import model_selection

    data.train["kfold"] = -1

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # TODO: stratify by both label and file_extension
    for f, (t_, v_) in enumerate(kf.split(X=data.train, y=data.train.label)):
        data.train.loc[v_, 'kfold'] = f

    print(data.train.sample(3))

    data.train.to_csv(f"{data.input_path}/train_folds.csv", index=False)
