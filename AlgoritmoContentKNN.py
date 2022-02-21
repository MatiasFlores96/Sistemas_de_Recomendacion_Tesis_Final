from surprise import AlgoBase
from surprise import PredictionImpossible
from DatasetFinal import DatasetFinal
import math
import numpy as np
import heapq

class AlgoritmoContentKNN(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # Se realiza una matriz de similitud basada en los atributos de los items
        # Se cargan vectores de generos y años para cada pelicula
        dataset = DatasetFinal()
        genres = dataset.ObtenerGeneros()
        years = dataset.ObtenerAnos()
        
        print("Realizando Matriz de Similitud...")
            
        # Se computa la distancia de los generos y años para cada combinacion de peliculas en una matriz 2x2
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for thisRating in range(self.trainset.n_items):
            if (thisRating % 100 == 0):
                print(thisRating, " de ", self.trainset.n_items)
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisMovieID = int(self.trainset.to_raw_iid(thisRating))
                otherMovieID = int(self.trainset.to_raw_iid(otherRating))
                genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                self.similarities[thisRating, otherRating] = genreSimilarity * yearSimilarity
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
                
        print("...Hecho.")
                
        return self
    
    def computeGenreSimilarity(self, movie1, movie2, genres):
        #Se realiza la similitud entre los genereos de dos peliculas
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def computeYearSimilarity(self, movie1, movie2, years):
        #Se realiza la similitud entre años de dos peliculas
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff / 10.0)
        return sim
    

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('Usuario o Item desconocido.')
        
        # Se crean valores de similitud entre este item y los otros que el usuario haya calificado
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i,rating[0]]
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
    