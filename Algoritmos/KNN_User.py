from CargadorDataset import CargadorDataset
from surprise import KNNBasic
from surprise.model_selection import GridSearchCV
from Evaluador import Evaluador
from Recomendador import Recomendador

import random
import numpy as np

def CargarDatos():
    #Se carga el dataset para poder enviar al evaluador
    dataset = CargadorDataset()
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
#KNN_param_grid = {'k': [10, 20, 40, 50],
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
print("Mejor valor de CalcularRMSE en Entrenamiento: ", KNN_grid_search.best_score['rmse'])
print("Mejores hiperparametros utilizados: ", KNN_grid_search.best_params['rmse'])




#KNN Neighborhood Based
KNN_params = KNN_grid_search.best_params['rmse']
KNN_best = KNNBasic(k=KNN_params['k'],
               sim_options=KNN_params['sim_options']
               )
# Construccion de Evaluador para evaluar cada Algoritmo
evaluador = Evaluador(datosEvaluacion, rankings)
evaluador.AgregarAlgoritmo(KNN_best, "KNN User-Based")
#Evaluacion de los Sistemas de Recomendacion realizados
evaluador.Evaluar(ranking=True, caracteristicas=True)

rankingsRec = []
#Construccion de Recomendador para realizar recomendaciones a un usuario
recomendador = Recomendador(datosEvaluacion, rankings)
recomendador.AgregarAlgoritmo(KNN_best, "KNN_User")
#Llamada a la funcion Recomendar. Se le pasa el dataset, Id del usuario y tamaño de recomendaciones
recomendador.Recomendar(dataset, 500, 10)


