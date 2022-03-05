from DatasetFinal import DatasetFinal
from AlgoritmoContentKNN import AlgoritmoContentKNN
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

# Construccion de Evaluador para evaluar cada Algoritmo
evaluador = Evaluador(datosEvaluacion, rankings)

contentKNN = AlgoritmoContentKNN(30)
evaluador.AgregarAlgoritmo(contentKNN, "ContentKNN")

#Evaluacion de los Sistemas de Recomendacion realizados
evaluador.Evaluar(rank=True, caracteristicas=True)


