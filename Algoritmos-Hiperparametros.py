# -*- coding: utf-8 -*-

from DatasetFinal import DatasetFinal
from ContentKNNAlgorithm import ContentKNNAlgorithm
from surprise import SVD 
from surprise import NMF 
from surprise import SVDpp
from surprise import KNNBasic
from surprise.model_selection import GridSearchCV
from Evaluador import Evaluador
from surprise import NormalPredictor

import random
import numpy as np

def LoadMovieLensData():
    ml = DatasetFinal()
    print("Loading movie ratings...")
    data = ml.CargarDataset()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.ObtenerRankingPopularidad()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

#Funk-SVD
#Preparacion de Modelo SVD
#SVD_param_grid = {'n_epochs': [25, 50, 75], 'lr_all': [0.005, 0.01],
#              'n_factors': [50, 100, 150], 'reg_all': [0.1, 0.5, 1]}

#Utilice los mejores para que no tarde tanto
SVD_param_grid = {'n_epochs': [75], 'lr_all': [0.01],
              'n_factors': [150], 'reg_all': [0.1]}


SVD_gs = GridSearchCV(SVD, SVD_param_grid, measures=['rmse', 'mae'], cv=3)

#Entrenamiento de Modelo SVD
#print("Modelo Funk-SVD")
#SVD_gs.fit(evaluationData)
#print("Mejor valor de RMSE en Entrenamiento: ", SVD_gs.best_score['rmse'])
#print("Mejores hiperparametros utilizados: ", SVD_gs.best_params['rmse'])

       
#NMF
#NMF_param_grid = {'n_epochs': [50, 75], 'lr_bu': [0.005, 0.01],
#              'n_factors': [100, 150], 'reg_pu': [0.1, 0.5]}

#NMF_param_grid = {'n_epochs': [50], 'lr_bu': [0.01],
#              'n_factors': [150], 'reg_pu': [0.5]}

#NMF_gs = GridSearchCV(NMF, NMF_param_grid, measures=['rmse', 'mae'], cv=3)

#Entrenamiento de Modelo NMF
#print("Modelo NMF")
#NMF_gs.fit(evaluationData)
#print("Mejor valor de RMSE en Entrenamiento: ", NMF_gs.best_score['rmse'])
#print("Mejores hiperparametros utilizados: ", NMF_gs.best_params['rmse'])

#PMF
#PMF_param_grid = {'n_epochs': [25, 50, 75], 'lr_all': [0.005, 0.01],
#              'n_factors': [50, 100, 150], 'reg_all': [0.1, 0.5, 1], 'biased':[False]}

#PMF_param_grid = {'n_epochs': [75], 'lr_all': [0.01],
#              'n_factors': [50], 'reg_all': [0.1]}

#PMF_gs = GridSearchCV(SVD, PMF_param_grid, measures=['rmse', 'mae'], cv=3)

#Entrenamiento de Modelo PMF
#print("Modelo PMF")
#PMF_gs.fit(evaluationData)
#print("Mejor valor de RMSE en Entrenamiento: ", PMF_gs.best_score['rmse'])
#print("Mejores hiperparametros utilizados: ", PMF_gs.best_params['rmse'])

#SVD++
#SVDpp_param_grid = {'n_epochs': [25, 50, 75], 'lr_all': [0.005, 0.01],
#              'n_factors': [50, 100, 150], 'reg_all': [0.1, 0.5, 1]}

#SVDpp_param_grid = {'n_epochs': [75], 'lr_all': [0.01],
#              'n_factors': [50], 'reg_all': [0.1]}

#SVDpp_gs = GridSearchCV(SVDpp, SVDpp_param_grid, measures=['rmse', 'mae'], cv=3)

#Entrenamiento de Modelo SVDpp
#print("Modelo SVDpp")
#SVDpp_gs.fit(evaluationData)
#print("Mejor valor de RMSE en Entrenamiento: ", SVDpp_gs.best_score['rmse'])
#print("Mejores hiperparametros utilizados: ", SVDpp_gs.best_params['rmse'])

#KNN Neighborhood based
#KNN_param_grid = {'k': [10, 20, 50],'sim_options': {'nombre': ['pearson', 'cosine'], 'user_based': [True]}}
#KNN_grid_search = GridSearchCV(KNNBasic, KNN_param_grid, measures=['rmse','mae'], cv=3, n_jobs = -1)

#print("Modelo KNN")

#KNN_grid_search.fit(evaluationData)
#print("Mejor valor de RMSE en Entrenamiento: ", KNN_grid_search.best_score['rmse'])
#print("Mejores hiperparametros utilizados: ", KNN_grid_search.best_params['rmse'])

#KNN Neighborhood based
#KNN_item_param_grid = {'k': [10, 20, 50],'sim_options': {'nombre': ['pearson', 'cosine'], 'user_based': [False]}}
#KNN_item_grid_search = GridSearchCV(KNNBasic, KNN_item_param_grid, measures=['rmse','mae'], cv=3)

#print("Modelo KNN-Item")

#KNN_item_grid_search.fit(evaluationData)
#print("Mejor valor de RMSE en Entrenamiento: ", KNN_item_grid_search.best_score['rmse'])
#print("Mejores hiperparametros utilizados: ", KNN_item_grid_search.best_params['rmse'])


#Content Based

# Construccion de Evaluador para evaluar cada Algoritmo
evaluator = Evaluator(evaluationData, rankings)

contentKNN = ContentKNNAlgorithm()
evaluator.AgregarAlgoritmo(contentKNN, "ContentKNN")

#Matrix Factorization Funk-SVD
#SVD_params = SVD_gs.best_params['rmse']
#SVDtuned = SVD(n_epochs = SVD_params['n_epochs'], lr_all = SVD_params['lr_all'], n_factors = SVD_params['n_factors'], reg_all = SVD_params['reg_all'])
#evaluator.AddAlgorithm(SVDtuned, "Funk-SVD - Entrenado")

#Matrix Factorization Funk-SVD sin entrenamiento
#SVDUntuned = SVD()
#evaluator.AddAlgorithm(SVDUntuned, "Funk-SVD - Sin Entrenar")

#Matrix Factorization NMF
#NMF_params = NMF_gs.best_params['rmse']
#NMFtuned = NMF(n_epochs = NMF_params['n_epochs'], lr_bu = NMF_params['lr_bu'], n_factors = NMF_params['n_factors'], reg_pu = NMF_params['reg_pu'])
#evaluator.AddAlgorithm(NMFtuned, "NMF")

#Matrix Factorization Funk-SVD
#PMF_params = PMF_gs.best_params['rmse']
#PMF = SVD(n_epochs = PMF_params['n_epochs'], lr_all = PMF_params['lr_all'], n_factors = PMF_params['n_factors'], reg_all = PMF_params['reg_all'], biased = PMF_params['biased'])
#evaluator.AddAlgorithm(PMF, "PMF")

#Matrix Factorization Funk-SVD
#SVDpp_params = SVDpp_gs.best_params['rmse']
#SVDpp = SVDpp(n_epochs = SVDpp_params['n_epochs'], lr_all = SVDpp_params['lr_all'], n_factors = SVDpp_params['n_factors'], reg_all = SVDpp_params['reg_all'])
#evaluator.AddAlgorithm(SVDpp, "SVDpp")

#Matrix Factorization Funk-SVD
#SVDpp_params = SVDpp_gs.best_params['rmse']
#SVDpp = SVDpp(n_epochs = SVDpp_params['n_epochs'], lr_all = SVDpp_params['lr_all'], n_factors = SVDpp_params['n_factors'], reg_all = SVDpp_params['reg_all'])
#evaluator.AddAlgorithm(SVDpp, "SVDpp")

#KNN Neighborhood Based
#KNN_params = KNN_grid_search.best_params['rmse']
#KNN = KNNBasic(k = KNN_params['k'], sim_options = KNN_params['sim_options'])
#evaluator.AddAlgorithm(KNN, "KNN User-Based")

#KNN Neighborhood Based
#KNN_item_params = KNN_item_grid_search.best_params['rmse']
#KNN_item = KNNBasic(k = KNN_item_params['k'], sim_options = KNN_item_params['sim_options'])
#evaluator.AddAlgorithm(KNN_item, "KNN Item-Based")

# Modelo aleatorio
Random = NormalPredictor()
evaluator.AgregarAlgoritmo(Random, "Random")

#Evaluacion de los Sistemas de Recomendacion realizados
evaluator.Evaluar(True)

#evaluator.SampleTopNRecs(ml)


