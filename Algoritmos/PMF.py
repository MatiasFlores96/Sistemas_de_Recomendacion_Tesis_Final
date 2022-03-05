from DatasetFinal import DatasetFinal
from surprise import SVD 
from surprise.model_selection import GridSearchCV
from Evaluador import Evaluador

import random
import numpy as np

def CargarDatos():
    #Se carga el dataset para poder enviar al evaluador
    dataset = DatasetFinal()
    datosEvaluacion = dataset.CargarDataset()
    #Este es para calcular la innovacion
    rankings = dataset.ObtenerRankingPopularidad()
    return (dataset, datosEvaluacion, rankings)

np.random.seed(0)
random.seed(0)

# Carga los datos para los algoritmos recomendadores.
# El dataset es para hacer un ejemplo de recomendacion.
(dataset, datosEvaluacion, rankings) = CargarDatos()
#PMF
#PMF_param_grid = {'n_epochs': [25, 50, 75],
#                  'lr_all': [0.005, 0.01],
#                  'n_factors': [50, 100, 150],
#                  'reg_all': [0.1, 0.5, 1],
#                  'biased': [False]
#                  }

#MEJORES PARAMETROS
PMF_param_grid = {'n_epochs': [75],
                  'lr_all': [0.01],
                  'n_factors': [50],
                  'reg_all': [0.1],
                  'biased': [False]}

PMF_gs = GridSearchCV(SVD, PMF_param_grid, measures=['rmse', 'mae'], cv=5)

#Entrenamiento de Modelo PMF
print("Modelo PMF")
PMF_gs.fit(datosEvaluacion)
print("Mejor valor de RMSE en Entrenamiento: ", PMF_gs.best_score['rmse'])
print("Mejores hiperparametros utilizados: ", PMF_gs.best_params['rmse'])

# Construccion de Evaluador para evaluar cada Algoritmo
evaluador = Evaluador(datosEvaluacion, rankings)

#Matrix Factorization PMF
PMF_params = PMF_gs.best_params['rmse']
PMF = SVD(n_epochs = PMF_params['n_epochs'],
          lr_all = PMF_params['lr_all'],
          n_factors = PMF_params['n_factors'],
          reg_all = PMF_params['reg_all'],
          biased = PMF_params['biased'])

evaluador.AgregarAlgoritmo(PMF, "PMF")

#Evaluacion de los Sistemas de Recomendacion realizados
evaluador.Evaluar(rank=True, caracteristicas=True)

#evaluador.SampleTopNRecs(ml)


