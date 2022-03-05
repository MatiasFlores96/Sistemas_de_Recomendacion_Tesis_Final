from surprise import AlgoBase
from surprise import PredictionImpossible
from DatasetFinal import DatasetFinal
import math
import numpy as np
import heapq

class AlgoritmoContentKNN(AlgoBase):

    def __init__(self, k):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # Se realiza una matriz de similitud basada en los atributos de los items
        # Se cargan vectores de generos y años para cada pelicula
        dataset = DatasetFinal()
        generos = dataset.ObtenerGeneros()
        years = dataset.ObtenerYears()
        
        print("Realizando Matriz de Similitud...")
            
        # Se computa la distancia de los generos y años para cada combinacion de peliculas en una matriz 2x2
        self.similitudes = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for ratingActual in range(self.trainset.n_items):
            if ratingActual % 100 == 0:
                print(ratingActual, " de ", self.trainset.n_items)
            for otroRating in range(ratingActual+1, self.trainset.n_items):
                movieIDActual = int(self.trainset.to_raw_iid(ratingActual))
                otroMovieID = int(self.trainset.to_raw_iid(otroRating))
                genreSimilarity = self.CalcularSimilitudGeneros(movieIDActual, otroMovieID, generos)
                yearSimilarity = self.CalcularSimilitudYears(movieIDActual, otroMovieID, years)
                self.similitudes[ratingActual, otroRating] = genreSimilarity * yearSimilarity
                self.similitudes[otroRating, ratingActual] = self.similitudes[ratingActual, otroRating]
                
        print("...Hecho.")
                
        return self
    
    def CalcularSimilitudGeneros(self, pelicula1, pelicula2, generos):
        #Se realiza la similitud entre los genereos de dos peliculas
        generos1 = generos[pelicula1]
        generos2 = generos[pelicula2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(generos1)):
            x = generos1[i]
            y = generos2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def CalcularSimilitudYears(self, pelicula1, pelicula2, years):
        #Se realiza la similitud entre años de dos peliculas
        diff = abs(years[pelicula1] - years[pelicula2])
        sim = math.exp(-diff / 10.0)
        return sim
    

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('Usuario o Item desconocido.')
        
        # Se crean valores de similitud entre este item y los otros que el usuario haya calificado
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similitudes[i,rating[0]]
            neighbors.append( (genreSimilarity, rating[1]) )
        
        # Se extraen los top-k ratings mas similares
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        # Se calcula el valor de similitud promedio de K vecinos ponderado por los ratings del usuario
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
            
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
    