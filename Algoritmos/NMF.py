from DatasetFinal import DatasetFinal
from surprise import NMF 
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

#NMF
#Este param grid es el que se utiliza para buscar los mejores parametros.
#Se encuentra comentado porque consume mucha memoria
#NMF_param_grid = {'n_epochs': [35, 50, 75],
#                  'lr_bu': [0.005, 0.01, 0.1],
#                  'n_factors': [50, 100, 150],
#                  'reg_pu': [0.1, 0.5]
#                  }

#MEJORES PARAMETROS
NMF_param_grid = {'n_epochs': [50],
                  'lr_bu': [0.01],
                  'n_factors': [150],
                  'reg_pu': [0.5]
                  }

NMF_gs = GridSearchCV(NMF, NMF_param_grid, measures=['rmse', 'mae'], cv=5)

#Entrenamiento de Modelo NMF
print("Modelo NMF")
NMF_gs.fit(datosEvaluacion)
print("Mejor valor de RMSE en Entrenamiento: ", NMF_gs.best_score['rmse'])
print("Mejores hiperparametros utilizados: ", NMF_gs.best_params['rmse'])

#Construccion de Evaluador para evaluar cada Algoritmo
evaluador = Evaluador(datosEvaluacion, rankings)

#Matrix Factorization NMF
NMF_params = NMF_gs.best_params['rmse']
NMFtuned = NMF(n_epochs=NMF_params['n_epochs'],
               lr_bu=NMF_params['lr_bu'],
               n_factors=NMF_params['n_factors'],
               reg_pu=NMF_params['reg_pu'])

evaluador.AgregarAlgoritmo(NMFtuned, "NMF")

#Evaluacion de los Sistemas de Recomendacion realizados
evaluador.Evaluar(rank=False, caracteristicas=True)

#evaluador.SampleTopNRecs(ml)


