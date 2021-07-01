import xlearn as xl


if __name__ == '__main__':
    ffm_model = xl.create_ffm()
    ffm_model.setTrain('./datasets/small_train.txt')
    ffm_model.setValidate('./datasets/small_test.txt')
    param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002, 'metric': 'acc', 'k': 5,  'stop_window': 3, 'epoch': 30, 'nthread': 6}
    ffm_model.fit(param, './model_criteo.out')
    ffm_model.setTest('./datasets/small_test.txt')
    ffm_model.setSigmoid()
    ffm_model.predict('./model_criteo.out', './ouput_criteo.txt')