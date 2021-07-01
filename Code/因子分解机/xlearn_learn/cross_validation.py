import xlearn as xl

ffm_model = xl.create_ffm()
ffm_model.setTrain('./small_train.txt')
# ffm_model.disableLockFree()
#ffm_model.disableNorm()
params = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002, 'metric': 'acc', 'opt': 'sgd', 'k': 8, 'epoch': 20}
ffm_model.cv(params)