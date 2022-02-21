from DatasetFinal import DatasetFinal
from surprise import KNNBasic
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

#KNN Neighborhood based
#KNN_param_grid = {'k': [10, 20, 50],
#                  'sim_options': {'name': ['pearson', 'cosine'],
#                                  'user_based': [True]
#                                  }
#                  }

KNN_param_grid = {'k': [50],
                  'sim_options': {'name': ['cosine'],
                                  'user_based': [True]
                                  }
                  }

KNN_grid_search = GridSearchCV(KNNBasic, KNN_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)

print("Modelo KNN")

KNN_grid_search.fit(datosEvaluacion)
print("Mejor valor de RMSE en Entrenamiento: ", KNN_grid_search.best_score['rmse'])
print("Mejores hiperparametros utilizados: ", KNN_grid_search.best_params['rmse'])


# Construccion de Evaluador para evaluar cada Algoritmo
evaluador = Evaluador(datosEvaluacion, rankings)


#KNN Neighborhood Based
KNN_params = KNN_grid_search.best_params['rmse']
KNN = KNNBasic(k=KNN_params['k'],
               sim_options=KNN_params['sim_options']
               )

evaluador.AgregarAlgoritmo(KNN, "KNN User-Based")

#Evaluacion de los Sistemas de Recomendacion realizados
evaluador.Evaluar(rank=False, caracteristicas=True)

#evaluador.SampleTopNRecs(ml)


