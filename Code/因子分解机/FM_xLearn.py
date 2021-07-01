import xlearn as xl


if __name__ == '__main__':
    fm_model = xl.create_fm()
    fm_model.setTrain('./datasets/titanic_train.txt')
    # fm_model.setValidate('./datasets/titanic_test.txt')
    param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002, 'metric': 'acc', 'k': 5}
    fm_model.cv(param)
    # fm_model.setTest('./datasets/titanic_test.txt')
    # fm_model.setSigmoid()
    # fm_model.predict('./model_titanic.out', './ouput_titanic.txt')