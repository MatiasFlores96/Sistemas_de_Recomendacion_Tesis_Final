from CargadorDataset import CargadorDataset
from Evaluador import Evaluador
from Recomendador import Recomendador
from surprise import NormalPredictor

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

#Modelo aleatorio
Random = NormalPredictor()

#Construccion de Evaluador para evaluar cada Algoritmo
evaluador = Evaluador(datosEvaluacion, rankings)
evaluador.AgregarAlgoritmo(Random, "Random")
#Evaluacion de los Sistemas de Recomendacion realizados
evaluador.Evaluar(ranking=True, caracteristicas=True)

rankingsRec = []
#Construccion de Recomendador para realizar recomendaciones a un usuario
recomendador = Recomendador(datosEvaluacion, rankings)
recomendador.AgregarAlgoritmo(Random, "Random")
#Llamada a la funcion Recomendar. Se le pasa el dataset, Id del usuario y tamaño de recomendaciones
recomendador.Recomendar(dataset, 500, 10)

