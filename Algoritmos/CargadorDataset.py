import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader
from collections import defaultdict

class CargadorDataset:
    ID_a_nombre = {}
    nombre_a_ID = {}
    ratingsPath = '../ml-latest-small/ratings.csv'
    moviesPath = '../ml-latest-small/movies.csv'
    
    #Esta funcion Carga los datasets desde la ruta donde se encuentran
    #Los datasets estan en csv y utiliza la duncion Reader de surprise
    #Para que queden de forma que la libreria pueda interpretarlos
    def CargarDataset(self):
        os.chdir(os.path.dirname(sys.argv[0]))
        ratingsDataset = 0
        self.ID_a_nombre = {}
        self.nombre_a_ID = {}
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)
                for fila in movieReader:
                    IDPelicula = int(fila[0])
                    nombrePelicula = fila[1]
                    self.ID_a_nombre[IDPelicula] = nombrePelicula
                    self.nombre_a_ID[nombrePelicula] = IDPelicula

        return ratingsDataset

    #Esta funcion devuelve los rankings de las peliculas
    #Se basa en que tan popular es una pelicula segun su cantidad de ratings recibidos
    def ObtenerRankingPopularidad(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)

        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for fila in ratingReader:
                IDPelicula = int(fila[1])
                ratings[IDPelicula] += 1
        rank = 1

        for IDPelicula, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[IDPelicula] = rank
            rank += 1

        return rankings

   #Esta funcion retorna un diccionario con los generos de las peliculas
    def ObtenerGeneros(self):
        generos = defaultdict(list)
        IDsGeneros = {}
        MaximoIDGenero = 0

        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)

            for fila in movieReader:
                IDPelicula = int(fila[0])
                listaGeneros = fila[2].split('|')
                listaIDsGeneros = []
                for genero in listaGeneros:

                    if genero in IDsGeneros:
                        IDGenero = IDsGeneros[genero]

                    else:
                        IDGenero = MaximoIDGenero
                        IDsGeneros[genero] = IDGenero
                        MaximoIDGenero += 1

                    listaIDsGeneros.append(IDGenero)
                generos[IDPelicula] = listaIDsGeneros

        for (IDPelicula, listaIDsGeneros) in generos.items():
            campo = [0] * MaximoIDGenero
            for IDGenero in listaIDsGeneros:
                campo[IDGenero] = 1
            generos[IDPelicula] = campo
        
        return generos

    #Esta funcion devuelve un diccionario con los años de las peliculas
    #La palabra Year o Years se utilizan en ingles para no utilizar ñ en los codigos
    def ObtenerYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)

        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)

            for fila in movieReader:
                IDPelicula = int(fila[0])
                titulo = fila[1]
                m = p.search(titulo)
                year = m.group(1)

                if year:
                    years[IDPelicula] = int(year)

        return years

    #Esta funcion devuelve el titulo del Item de acuerdo al Id pasado
    def ObtenerNombreItem(self, IDPelicula):

        if IDPelicula in self.ID_a_nombre:
            return self.ID_a_nombre[IDPelicula]

        else:
            return ""

    #Esta funcion devuelve el id del Item de acuerdo al nombre pasado
    def ObtenerIDItem(self, nombrePelicula):

        if nombrePelicula in self.nombre_a_ID:
            return self.nombre_a_ID[nombrePelicula]

        else:
            return 0