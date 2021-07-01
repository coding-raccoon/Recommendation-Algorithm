import xlearn as xl

ffm_model = xl.create_ffm()
ffm_model.setTrain("./small_train.txt")
param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002}
ffm_model.fit(param, "./model.out")
ffm_model.setTest("./small_test.txt")
ffm_model.predict('./model.out', './ouput.txt')