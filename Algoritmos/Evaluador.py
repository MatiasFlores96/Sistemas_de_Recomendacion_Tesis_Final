from GeneradorConjuntos import GeneradorConjuntos
from AnalizadorAlgoritmo import AnalizadorAlgoritmo


class Evaluador:
    #Se crea un array de algoritmos donde se guardaria cada uno para utilizar mas adelante
    algoritmos = []

    #Se inicializan los datos que se van a utilizar mas adelante
    #datos trae las inicializaciones hecha en la clase EvaluacionDatos importada
    def __init__(self, dataset, rankings):
        datos = GeneradorConjuntos(dataset, rankings)
        self.dataset = datos

    #Esta funcion inicializa los algoritmos en base de EvaliacionAlgoritmo
    #En este caso almacenaria los algoritmos en el array
    #Como consumen mucho procesamiento cada uno, se decidio hacerlo individualmente
    #Existe la posibilidad de agregar distintos algoritmos de cualquier manera
    def AgregarAlgoritmo(self, algoritmo, nombre):
        algo = AnalizadorAlgoritmo(algoritmo, nombre)
        self.algoritmos.append(algo)

    #Esta funcion se encarga de evaluar cada uno de los algoritmos que se pasen
    def Evaluar(self, ranking, caracteristicas):
        # ranking y caracteristicas son booleanos que se utilizan para determinar que metricas utilizar
        # Las metricas de exactitud siempre se realizan
        resultados = {}
        # Se fija en cada algoritmo que le pasan, y llama a la funcion evaluar de la clase EvaluacionAlgoritmo
        for algoritmo in self.algoritmos:
            print("Evaluando ", algoritmo.ObtenerNombre(), "...")
            #Guarda los resultados obtenidos en el array para mostrar por pantalla posteriormente
            resultados[algoritmo.ObtenerNombre()] = algoritmo.Evaluar(self.dataset, ranking, caracteristicas)

        print("\nExactitud:")
        print("{:<10} {:<10} {:<10} {:<10} {:<10} ".format("Algoritmo",
                                                           "RMSE",
                                                           "MAE",
                                                           "MSE",
                                                           "FCP"
                                                           ))

        for (nombre, metricas) in resultados.items():
            print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(nombre,
                                                                          metricas["RMSE"],
                                                                          metricas["MAE"],
                                                                          metricas["MSE"],
                                                                          metricas["FCP"]
                                                                          ))
        if ranking:
            print("\nPrecision:")
            print("{:<12} {:<12} {:<12} {:<12} {:<12} "
                  "{:<12} {:<12} {:<12} {:<12} {:<12}".format("Precision@1",
                                                              "Precision@2",
                                                              "Precision@3",
                                                              "Precision@4",
                                                              "Precision@5",
                                                              "Precision@6",
                                                              "Precision@7",
                                                              "Precision@8",
                                                              "Precision@9",
                                                              "Precision@10"))
            for (nombre, metricas) in resultados.items():
                print(
                    "{:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} "
                    "{:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(metricas["Pre-1"],
                                                                               metricas["Pre-2"],
                                                                               metricas["Pre-3"],
                                                                               metricas["Pre-4"],
                                                                               metricas["Pre-5"],
                                                                               metricas["Pre-6"],
                                                                               metricas["Pre-7"],
                                                                               metricas["Pre-8"],
                                                                               metricas["Pre-9"],
                                                                               metricas["Pre-10"]))

            print("\nRecall:")
            print("{:<10} {:<10} {:<10} {:<10} {:<10} "
                  "{:<10} {:<10} {:<10} {:<10} {:<10}".format("Recall@1",
                                                              "Recall@2",
                                                              "Recall@3",
                                                              "Recall@4",
                                                              "Recall@5",
                                                              "Recall@6",
                                                              "Recall@7",
                                                              "Recall@8",
                                                              "Recall@9",
                                                              "Recall@10"))

            for (nombre, metricas) in resultados.items():
                print(
                    "{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} "
                    "{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(metricas["Rec-1"],
                                                                               metricas["Rec-2"],
                                                                               metricas["Rec-3"],
                                                                               metricas["Rec-4"],
                                                                               metricas["Rec-5"],
                                                                               metricas["Rec-6"],
                                                                               metricas["Rec-7"],
                                                                               metricas["Rec-8"],
                                                                               metricas["Rec-9"],
                                                                               metricas["Rec-10"]))

            print("\nF1:")
            print("{:<10} {:<10} {:<10} {:<10} {:<10} "
                  "{:<10} {:<10} {:<10} {:<10} {:<10}".format("F1@1",
                                                              "F1@2",
                                                              "F1@3",
                                                              "F1@4",
                                                              "F1@5",
                                                              "F1@6",
                                                              "F1@7",
                                                              "F1@8",
                                                              "F1@9",
                                                              "F1@10"))

            for (nombre, metricas) in resultados.items():
                print(
                    "{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} "
                    "{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(metricas["F1-1"],
                                                                               metricas["F1-2"],
                                                                               metricas["F1-3"],
                                                                               metricas["F1-4"],
                                                                               metricas["F1-5"],
                                                                               metricas["F1-6"],
                                                                               metricas["F1-7"],
                                                                               metricas["F1-8"],
                                                                               metricas["F1-9"],
                                                                               metricas["F1-10"]))

        if caracteristicas:
            print("\nCaracteristicas:")
            print("{:<10} {:<10} {:<10}".format("Cobertura",
                                                "Diversidad",
                                                "Innovacion"
                                                ))

            for (nombre, metricas) in resultados.items():
                print("{:<10.4f} {:<10.4f} {:<10.4f} ".format(metricas["Cobertura"],
                                                              metricas["Diversidad"],
                                                              metricas["Innovacion"]
                                                              ))

        print("\nLeyenda:")
        print("RMSE: Root Mean Squared Error. Cuanto menor sea el valor mayor es la exactitud.")
        print("MAE: Mean Absolute Error. Cuanto menor sea el valor mayor es la exactitud.")
        print("MSE: Mean Squared Error. Cuanto menor sea el valor mayor es la exactitud.")
        print("FCP: Fraction of Concordant Pairs. Cuanto mayor sea el valor mayor es la exactitud.")
        if ranking:
            print("Precision: Proporcion de items recomendados que son relevantes")
            print("Recall: Proporcion de items relevantes que son recomendados")
            print("F1: Media harmonica entre Precision y Recall")
        if caracteristicas:
            print("Cobertura: Radio de usuarios para los cuales las "
                  "recomendaciones arriba de un limite existen. Mayor valor es mejor.")
            print("Diversidad: 1-S, donde S es el promedio de similitud entre cada par "
                  "de recomendaciones para un usuario. Mayor significa mas diverso")
            print("Innovacion: Rango promedio de popularidad de los items recomendados. "
                  "Cuanto mayor sea mas innovador")