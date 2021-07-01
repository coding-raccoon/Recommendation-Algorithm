from experiment.topnexperiment import TopNExperiment
from experiment.ratingexperiment import RatingPredExperiment
from dataload.movielens import MovieLensForRatingPred, MovieLensForTopN
from similarity.topnsimilarity import TopNSimilarity
from similarity.ratingsimilarity import RatingPredSimilarity
from model.cfmodels import UserCFTopN, ItemCFTopN, UserCFRating, ItemCFRating
from model.graphmodels import MaterialDiffusionTopN, HeatConductionTopN, HybridBiGraphTopN, PersonalRank
import numpy as np
from metrics.topnmetrics import TopNMetrics
from metrics.ratingmetrics import RatingPredMetrics

if __name__ == '__main__':
    # Sim = TopNSimilarity()
    M = 6
    '''
    实验：基于协同过滤的topN推荐算法
    '''
    # k_list = np.arange(5, 160, 5)
    # k_list = [5,6]
    # Model_Info = [["ucf_jaccard", UserCFTopN, (10, Sim.jaccard_sim), tuple([k_list])],
    #                ["ucf_cossine", UserCFTopN, (10, Sim.cossine_sim), tuple([k_list])],
    #               ["ucf_penalty_cosssine", UserCFTopN, (10, Sim.penalty_cossine_sim), tuple([k_list])],
    #               ["ucf_johnbreese", UserCFTopN, (10, Sim.johnbreese_sim), tuple([k_list])],
    #               ["icf_jaccard", ItemCFTopN, (10, Sim.jaccard_sim), tuple([k_list])],
    #               ["icf_cossine", ItemCFTopN, (10, Sim.cossine_sim), tuple([k_list])],
    #               ["icf_penalty_cosssine", ItemCFTopN, (10, Sim.penalty_cossine_sim), tuple([k_list])],
    #               ["icf_johnbreese", ItemCFTopN, (10, Sim.johnbreese_sim), tuple([k_list])]]
    '''
    实验：物质扩散算法和热传导算法
    '''
    # step_list = np.arange(1, 50)
    # Model_Info = [['hybridBiGraph', HybridBiGraphTopN, (10, 2), tuple([step_list])],
    #                ['material_diffusion', MaterialDiffusionTopN, (10, 2), tuple([step_list])],
    #                ['heat_conduction', HeatConductionTopN, (10, 2), tuple([step_list])]]
    '''
    实验：personal—rank算法
    '''
    # step_list = np.arange(1, 50)
    # Model_Info = [['persoanl_rank', PersonalRank, (10, 0.8), tuple([step_list])]]
    # dataloader = LoadData
    """
    实验：用于评分预测的协同过滤算法
    """
    Sim = RatingPredSimilarity()
    metrics = RatingPredMetrics()
    # model = UserCFRating(Sim.pearson_correlation, 5)
    # dataloader = MovieLensForRatingPred(M).data_iter()
    # for data_train, data_validation in dataloader:
    #     model = UserCFRating(Sim.pearson_correlation, 6000)
    #     pred = model.predict(data_train)
    #     true = data_validation
    #     print(pred)
    #     print(true)
    #     num = 0
    #     for i in range(true.shape[0]):
    #         for j in range(true.shape[1]):
    #             if true[i, j] != 0:
    #                 num += 1
    #                 print(i, j, true[i, j], round(pred[i, j]))
    #     print(metrics.rmse(true, pred))
    #     print(metrics.mae(true, pred))
    #     break
    k_list = np.arange(1, 20, 2)
    Model_Info = [["ucf_pearson", UserCFRating, (Sim.pearson_correlation, 0), tuple([k_list])]]
                  # ["ucf_cosine", UserCFRating, (Sim.cosine_sim, 0), tuple([k_list])],
                  # ["icf_cosine", ItemCFRating, (Sim.cosine_sim, 0), tuple([k_list])],
                  # ["icf_pearson", ItemCFRating, (Sim.pearson_correlation, 0), tuple([k_list])],
                  # ["icf_adjust_coine", ItemCFRating, (Sim.adjust_cosine_sim, 0), tuple([k_list])]]
    # print(cf_topn_experiment.Model_Info[1][-1][0])
    # print(cf_topn_experiment)
    dataloader = MovieLensForRatingPred
    rating_experiment = RatingPredExperiment(dataloader, M, Model_Info)
    rating_experiment.conduct()