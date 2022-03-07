from CargadorDataset import CargadorDataset
from surprise import SVD 
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

#Funk-SVD
#Busqueda de mejores hiperparametros para SVD
#SVD_param_grid = {'n_epochs': [25, 50, 75],
#                  'lr_all': [0.005, 0.01],
#                  'n_factors': [50, 100, 150],
#                  'reg_all': [0.1, 0.5, 1]
#                  }

#Utilice los mejores hiperparametros para que no tarde tanto
SVD_param_grid = {'n_epochs': [75],
                  'lr_all': [0.01],
                  'n_factors': [150],
                  'reg_all': [0.1]
                  }


SVD_gs = GridSearchCV(SVD, SVD_param_grid, measures=['rmse', 'mae'], cv=5)

#Entrenamiento de Modelo SVD
print("Modelo Funk-SVD")
SVD_gs.fit(datosEvaluacion)
print("Mejor valor de CalcularRMSE en Entrenamiento: ", SVD_gs.best_score['rmse'])
print("Mejores hiperparametros utilizados: ", SVD_gs.best_params['rmse'])

#Matrix Factorization Funk-SVD
SVD_params = SVD_gs.best_params['rmse']
SVD_best= SVD(n_epochs=SVD_params['n_epochs'],
               lr_all=SVD_params['lr_all'],
               n_factors=SVD_params['n_factors'],
               reg_all=SVD_params['reg_all']
               )

# Construccion de Evaluador para evaluar cada Algoritmo
evaluador = Evaluador(datosEvaluacion, rankings)
evaluador.AgregarAlgoritmo(SVD_best, "Funk-SVD")

#Evaluacion de los Sistemas de Recomendacion realizados
evaluador.Evaluar(ranking=True, caracteristicas=True)

rankingsRec = []
#Construccion de Recomendador para realizar recomendaciones a un usuario
recomendador = Recomendador(datosEvaluacion, rankings)
recomendador.AgregarAlgoritmo(SVD_best, "SVD_Funk")
#Llamada a la funcion Recomendar. Se le pasa el dataset, Id del usuario y tamaño de recomendaciones
recomendador.Recomendar(dataset, 500, 10)

