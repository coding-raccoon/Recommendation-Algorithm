from .base import BaseExperiment
import numpy as np
from matplotlib import pyplot as plt


class RatingPredExperiment(BaseExperiment):
    """
    TopN推荐模型试验流程，继承自试验基类
    """
    def __init__(self, dataloader, M, Model_Info, figsize=(8, 6)):
        """
        试验初始化
        :param dataloader: 数据加载器
        :param M: K交叉验证折数
        :param Model_Info: 眼进行试验的模型信息
        :param figsize: 最后展示阶段作图的大小
        """
        super(RatingPredExperiment, self).__init__(dataloader, M, Model_Info)
        self.mean_metrics = {}
        self.figsize = figsize

    def conduct(self):
        self.evaluate()
        for model_name in self.dict_metrics.keys():
            print("test algorithm: {}".format(model_name))
            self.mean_metrics[model_name] = {'mae': [], 'rmse': []}
            for metrics in self.mean_metrics[model_name].keys():
                for i in range(len(self.dict_metrics[model_name])):
                    self.mean_metrics[model_name][metrics].append(self.dict_metrics[model_name][i][metrics])
                self.mean_metrics[model_name][metrics] = np.mean(self.mean_metrics[model_name][metrics], axis=0)
        self.show_result()

    def show_result(self):
        """
        结果展示
        """
        fig_size = self.figsize
        num_models = len(self.Model_Info)
        plt.figure(figsize=(fig_size[0]*(num_models+1)/2, fig_size[1]*2))
        color_map = ['r', 'b', 'y', 'g', 'c', 'k', 'm', '#FF00FF']
        # 画出每个测试的模型，评估参数随超参数变化的情况
        for i, model_name in enumerate(self.dict_metrics.keys()):
            plt.subplot(2, (num_models+1)/2, i+1)
            plt.title('Metrcis Varies in Model {}'.format(model_name))
            for j, key in enumerate(['mae', 'rmse']):
                plt.plot(self.Model_Info[i][-1][0], self.mean_metrics[model_name][key], c=color_map[j], label=key)
            plt.xlabel("variable_hyperparam")
            plt.ylabel("metrics_value")
            plt.legend()
        plt.savefig('./figures/by_model.png')
        # 画出各个评估参数在不同模型中的变化情况
        plt.figure(figsize=(fig_size[0]*2, fig_size[1]*2))
        for i, metrics in enumerate(["mae", "rmse"]):
            plt.subplot(2, 2, i+1)
            plt.title("{} varies in all models".format(metrics))
            for j, model_name in enumerate(self.dict_metrics.keys()):
                plt.plot(self.Model_Info[j][-1][0], self.mean_metrics[model_name][metrics],
                         c=color_map[j], label=model_name)
            plt.xlabel("variable_hyperparam")
            plt.ylabel("metrcis_value")
            plt.legend()
        plt.savefig('./figures/by_metrics.png')