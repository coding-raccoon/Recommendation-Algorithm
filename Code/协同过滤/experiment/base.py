from abc import abstractmethod


class BaseExperiment(object):
    """
    实验测试方法基类，不同类型的实验均继承自这个类
    """
    def __init__(self, dataloader, M, Model_Info):
        """
        初始化一个测试实验
        :param dataloader: 数据加载方法
        :param M: 交叉验证折数
        :param Model_Info: 各个要测试的模型的信息列表，每个列表元素是以下形式的列表
                            [模型名（字符串），模型类名，模型初始化信息（元组），模型测试的超参变化（元组）]
        """
        super(BaseExperiment, self).__init__()
        self.dataloader = dataloader
        self.M = M
        self.Model_Info = Model_Info
        self.dict_metrics = {}    # dict，key值为模型名，value为对应的参数信息

    def evaluate(self):
        data_iter = self.dataloader(self.M).data_iter()
        iter_nums =0
        for data_train, data_validation in data_iter:
            iter_nums += 1
            print("Data_iter_nums: {}".format(iter_nums))
            print("--------------------------------------")
            for i in range(len(self.Model_Info)):
                model_name = self.Model_Info[i][0]
                print("model_name: {}".format(model_name))
                model = self.Model_Info[i][1](*self.Model_Info[i][2])         # 创建模型
                current_metrics = model.evaluate(data_train, data_validation, *self.Model_Info[i][3])    # 调用模型evaluate方法完成模型训练、预测、
                if model_name not in self.dict_metrics.keys():
                    self.dict_metrics[model_name] = []
                self.dict_metrics[model_name].append(current_metrics)
            break

    @abstractmethod
    def show_result(self):
        pass

    @abstractmethod
    def conduct(self):
        pass
