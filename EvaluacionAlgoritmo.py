from MetricasEvaluacion import MetricasEvaluacion


class EvaluacionAlgoritmo:
    def __init__(self, algoritmo, nombre):
        self.algoritmo = algoritmo
        self.nombre = nombre

    def Evaluar(self, datosEvaluacion, rank, caracteristicas):
        #Creamos un diccionario de metricas para guardar todos los resultados obtenidos
        #Los resultados son RMSE, MAE, MSE, FCP
        #Si rank es True: Precision, Recall y F1 para un determinado tamaño n
        #Si caracteristicas es True: Cobertura, Diversidad e Innovacion
        metricas = {}
        # Evaluacion de Exactitud
        print("Evaluando Exactitud...")
        #Se hace el entrenamiento del modelo con el algoritmo pasado
        #El algoritmo ya posee los mejores hiperparametros
        self.algoritmo.fit(datosEvaluacion.ObtenerTrainSet())
        predicciones = self.algoritmo.test(datosEvaluacion.ObtenerTestSet())
        print("Analizando RMSE, MAE, MSE & FCP...")
        metricas["RMSE"] = MetricasEvaluacion.RMSE(predicciones)
        metricas["MAE"] = MetricasEvaluacion.MAE(predicciones)
        metricas["MSE"] = MetricasEvaluacion.MSE(predicciones)
        metricas["FCP"] = MetricasEvaluacion.FCP(predicciones)

        if rank:
            pre, rec, F1 = "Pre-", "Rec-", "F1-"
            #El tope es para que termine el bucle
            #Cuanto mas grande sea el valor, mas mediciones hara de estas metricas
            #Para este caso frenamos en 10, ya que hacerlo mas consume mas procesamiento y
            #Los siguientes resultados seguian siendo muy parecidos a los ya obtenidos
            #Si se quiere utilizar otro tamaño, se debe modificar tambien en evaluador
            tope = 11

            #El siguiente bucle es para evaluar las metricas para cada tamaño de la prediccion
            for k in range(1, tope):
                precision, recall, F1_Score = MetricasEvaluacion.ResultadosRanking(predicciones, k, 3.5)
                print("Evaluando Precision, Recall y F1 para una lista de tamaño ", k)
                metricas[pre + str(k)] = precision
                metricas[rec + str(k)] = recall
                metricas[F1 + str(k)] = F1_Score

        # Evaluacion de caracteristicas del sistema de recomendacion
        if caracteristicas:
            print("Realizando recomendaciones con todo el dataset para analizar caracteristicas...")
            self.algoritmo.fit(datosEvaluacion.ObtenerTrainSetCompleto())
            prediccionesTotal = self.algoritmo.test(datosEvaluacion.ObtenerAntiTestsetCompleto())
            topNPredichos = MetricasEvaluacion.ObtenerTopN(prediccionesTotal, 10, 4.0)

            print("Analizando Cobertura...")
            # Mide la cobertura con un limite minimo de 4 de calificacion
            metricas["Cobertura"] = MetricasEvaluacion.Cobertura(topNPredichos,
                                                                 datosEvaluacion.ObtenerTrainSetCompleto().n_users,
                                                                 4.0)
            print("Analizando Diversidad...")
            # Mide la diversidad de las recomendaciones:
            metricas["Diversidad"] = MetricasEvaluacion.Diversidad(topNPredichos,
                                                                   datosEvaluacion.ObtenerSimilitudes())
            print("Analizando Innovacion...")
            # Mide la innovacion de las recomendaciones
            metricas["Innovacion"] = MetricasEvaluacion.Innovacion(topNPredichos,
                                                                   datosEvaluacion.ObtenerRankingsPopularidad())
        print("Analisis Completo")
        #Devuelve las metricas para que las muestre Evaluador
        return metricas

    def ObtenerNombre(self):
        return self.nombre

    def ObtenerAlgoritmo(self):
        return self.algoritmo
