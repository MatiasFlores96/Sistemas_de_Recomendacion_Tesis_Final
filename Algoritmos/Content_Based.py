from CargadorDataset import CargadorDataset
from AlgoritmoContent import AlgoritmoContent
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

contentKNN = AlgoritmoContent(30)

# Construccion de Evaluador para evaluar cada Algoritmo
evaluador = Evaluador(datosEvaluacion, rankings)
evaluador.AgregarAlgoritmo(contentKNN, "ContentKNN")
#Evaluacion de los Sistemas de Recomendacion realizados
evaluador.Evaluar(ranking=True, caracteristicas=True)

rankingsRec = []
#Construccion de Recomendador para realizar recomendaciones a un usuario
recomendador = Recomendador(datosEvaluacion, rankings)
recomendador.AgregarAlgoritmo(contentKNN, "ContentKNN")
#Llamada a la funcion Recomendar. Se le pasa el dataset, Id del usuario y tamaño de recomendaciones
recomendador.Recomendar(dataset, 500, 10)


