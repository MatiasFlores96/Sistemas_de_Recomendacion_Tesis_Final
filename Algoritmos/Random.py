from DatasetFinal import DatasetFinal
from Evaluador import Evaluador
from surprise import NormalPredictor

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

#Construccion de Evaluador para evaluar cada Algoritmo
evaluador = Evaluador(datosEvaluacion, rankings)

#Modelo aleatorio
Random = NormalPredictor()
evaluador.AgregarAlgoritmo(Random, "Random")

#Evaluacion de los Sistemas de Recomendacion realizados
evaluador.Evaluar(rank=False, caracteristicas=True)


