import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader
from collections import defaultdict

class DatasetFinal:
    movieID_to_name = {}
    name_to_movieID = {}
    ratingsPath = '../ml-latest-small/ratings.csv'
    moviesPath = '../ml-latest-small/movies.csv'
    
    #Esta funcion Carga los datasets desde la ruta donde se encuentran
    #Los datasets estan en csv y utiliza la duncion Reader de surprise
    #Para que queden de forma que la libreria pueda interpretarlos
    def CargarDataset(self):
        os.chdir(os.path.dirname(sys.argv[0]))
        ratingsDataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)
                for row in movieReader:
                    movieID = int(row[0])
                    movieName = row[1]
                    self.movieID_to_name[movieID] = movieName
                    self.name_to_movieID[movieName] = movieID

        return ratingsDataset

    #Esta funcion devuelve los rankings de las peliculas
    #Se basa en que tan popular es una pelicula segun su cantidad de ratings recibidos
    def ObtenerRankingPopularidad(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)

        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1
        rank = 1

        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1

        return rankings

   #Esta funcion retorna un diccionario con los generos de las peliculas
    def ObtenerGeneros(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0

        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  #Skip header line

            for row in movieReader:
                movieID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:

                    if genre in genreIDs:
                        genreID = genreIDs[genre]

                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1

                    genreIDList.append(genreID)
                genres[movieID] = genreIDList

        for (movieID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield            
        
        return genres

    #Esta funcion devuelve un diccionario con los años de las peliculas
    def ObtenerYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)

        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)

            for row in movieReader:
                movieID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)

                if year:
                    years[movieID] = int(year)

        return years

    #Esta funcion devuelve el titulo del Item de acuerdo al Id pasado
    def ObtenerNombreItem(self, movieID):

        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]

        else:
            return ""

    #Esta funcion devuelve el id del Item de acuerdo al nombre pasado
    def ObtenerIDItem(self, movieName):

        if movieName in self.name_to_movieID:
            return self.name_to_movieID[movieName]

        else:
            return 0