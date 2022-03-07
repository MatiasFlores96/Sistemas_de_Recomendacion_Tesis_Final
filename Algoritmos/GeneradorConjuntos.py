from surprise.model_selection import train_test_split
from surprise import KNNBaseline


class GeneradorConjuntos:
    def __init__(self, datos, rankingsPopularidad):
        # Creacion de un split entre set de entrenamiento y de prueba de 75/25
        self.trainSet, self.testSet = train_test_split(datos, test_size=.25, random_state=1)

        # Creacion de un set de entrenamiento para evaluacion
        self.trainSetCompleto = datos.build_full_trainset()

        # Creacion de un anti test set que recolecte todos los datos que no estan en el set de entrenamiento
        self.antiTestSetCompleto = self.trainSetCompleto.build_anti_testset()

        # Matriz de similitud entre items para medir la diversidad
        opciones = {'name': 'cosine', 'user_based': False}
        self.algoritmoSimilitud = KNNBaseline(sim_options=opciones)
        self.algoritmoSimilitud.fit(self.trainSetCompleto)

        # Creacion de rankings para medir la innovacion
        self.rankings = rankingsPopularidad

    def ObtenerTrainSetCompleto(self):
        return self.trainSetCompleto

    def ObtenerAntiTestsetCompleto(self):
        return self.antiTestSetCompleto

    def ObtenerSetAntiTestParaUsuario(self, usuario):
        trainSet = self.trainSetCompleto
        fill = trainSet.global_mean
        antiTestSet = []
        idUsuario = trainSet.to_inner_uid(str(usuario))
        itemsDeUsuarios = set([j for (j, _) in trainSet.ur[idUsuario]])
        antiTestSet += [(trainSet.to_raw_uid(idUsuario), trainSet.to_raw_iid(i), fill) for
                        i in trainSet.all_items() if
                        i not in itemsDeUsuarios]
        return antiTestSet

    def ObtenerTrainSet(self):
        return self.trainSet

    def ObtenerTestSet(self):
        return self.testSet

    def ObtenerSimilitudes(self):
        return self.algoritmoSimilitud

    def ObtenerRankingsPopularidad(self):
        return self.rankings
