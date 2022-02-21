from MetricasEvaluacion import MetricasEvaluacion

class EvaluacionAlgoritmo:
    def __init__(self, algoritmo, nombre):
        self.algoritmo = algoritmo
        self.nombre = nombre
        
    def Evaluar(self, datosEvaluacion, rank, caracteristicas, n=10):
        metricas = {}
        # Evaluacion de Exactitud
        print("Evaluando Exactitud...")
        self.algoritmo.fit(datosEvaluacion.GetTrainSet())
        predicciones = self.algoritmo.test(datosEvaluacion.GetTestSet())
        print("Analizando RMSE, MAE, MSE & FCP...")
        metricas["RMSE"] = MetricasEvaluacion.RMSE(predicciones)
        metricas["MAE"] = MetricasEvaluacion.MAE(predicciones)
        metricas["MSE"] = MetricasEvaluacion.MSE(predicciones)
        metricas["FCP"] = MetricasEvaluacion.FCP(predicciones)

        if (rank):
            Pre = "Pre-"
            Rec = "Rec-"
            F1 = "F1-"
            tope = 11
            for k in range(1,tope):
                precision, recall, F1_Score = MetricasEvaluacion.ResultadosRanking(predicciones, k, 3.5)
                print("Evaluando Precision, Recall y F1 para una lista de tamaño ", k)
                metricas[Pre+str(k)] = precision
                metricas[Rec+str(k)] = recall
                metricas[F1+str(k)] = F1_Score

        #Evaluacion de caracteristicas del sistema de recomendacion
        if (caracteristicas):
            print("Realizando recomendaciones con todo el dataset para analizar caracteristicas...")
            self.algoritmo.fit(datosEvaluacion.GetFullTrainSet())
            prediccionesTotal = self.algoritmo.test(datosEvaluacion.GetFullAntiTestSet())
            topNPredichos = MetricasEvaluacion.ObtenerTopN(prediccionesTotal, n)
            print("Analizando Cobertura...")
            # Mide la cobertura con un limite minimo de 4 de ratings
            metricas["Cobertura"] = MetricasEvaluacion.Cobertura(topNPredichos, datosEvaluacion.GetFullTrainSet().n_users, ratingThreshold=4.0)
            print("Analizando Diversidad...")
            # Mide la diversidad de las recomendaciones:
            metricas["Diversidad"] = MetricasEvaluacion.Diversidad(topNPredichos, datosEvaluacion.GetSimilarities())
            print("Analizando Innovacion...")
            # Mide la innovacion de las recomendaciones
            metricas["Innovacion"] = MetricasEvaluacion.Innovacion(topNPredichos, datosEvaluacion.GetPopularityRankings())

        print("Analisis Completo")
        return metricas
    
    def ObtenerNombre(self):
        return self.nombre
    
    def ObtenerAlgoritmo(self):
        return self.algoritmo
    
    