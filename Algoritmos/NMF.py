from CargadorDataset import CargadorDataset
from surprise import NMF 
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
print("Mejor valor de CalcularRMSE en Entrenamiento: ", NMF_gs.best_score['rmse'])
print("Mejores hiperparametros utilizados: ", NMF_gs.best_params['rmse'])

#Matrix Factorization NMF
NMF_params = NMF_gs.best_params['rmse']
NMF_best = NMF(n_epochs=NMF_params['n_epochs'],
               lr_bu=NMF_params['lr_bu'],
               n_factors=NMF_params['n_factors'],
               reg_pu=NMF_params['reg_pu'])

#Construccion de Evaluador para evaluar cada Algoritmo
evaluador = Evaluador(datosEvaluacion, rankings)
evaluador.AgregarAlgoritmo(NMF_best, "NMF")
#Evaluacion de los Sistemas de Recomendacion realizados
evaluador.Evaluar(ranking=True, caracteristicas=True)

rankingsRec = []
#Construccion de Recomendador para realizar recomendaciones a un usuario
recomendador = Recomendador(datosEvaluacion, rankings)
recomendador.AgregarAlgoritmo(NMF_best, "NMF")
#Llamada a la funcion Recomendar. Se le pasa el dataset, Id del usuario y tama??o de recomendaciones
recomendador.Recomendar(dataset, 500, 10)
