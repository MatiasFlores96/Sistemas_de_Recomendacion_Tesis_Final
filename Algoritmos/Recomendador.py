from GeneradorConjuntos import GeneradorConjuntos
from AnalizadorAlgoritmo import AnalizadorAlgoritmo


class Recomendador:
    algoritmos = []

    def __init__(self, dataset, rankings):
        datos = GeneradorConjuntos(dataset, rankings)
        self.dataset = datos

    def AgregarAlgoritmo(self, algoritmo, nombre):
        algo = AnalizadorAlgoritmo(algoritmo, nombre)
        self.algoritmos.append(algo)

    def Recomendar(self, datasetCompleto, usuario, k):

        for algo in self.algoritmos:
            print("\nRealizando recomendaciones con ", algo.ObtenerNombre())

            print("\nCreando Modelo de Recomendacion...")
            trainSet = self.dataset.ObtenerTrainSetCompleto()
            algo.ObtenerAlgoritmo().fit(trainSet)

            print("Realizando Recomendaciones...")
            testSet = self.dataset.ObtenerSetAntiTestParaUsuario(usuario)

            predicciones = algo.ObtenerAlgoritmo().test(testSet)

            recomendaciones = []

            print("\nRecomendaciones para el usuario ", usuario, ":")
            for userID, movieID, actualRating, estimatedRating, _ in predicciones:
                intMovieID = int(movieID)
                recomendaciones.append((intMovieID, estimatedRating))

            recomendaciones.sort(key=lambda x: x[1], reverse=True)

            for ratings in recomendaciones[:10]:
                print(datasetCompleto.ObtenerNombreItem(ratings[0]), ratings[1])


