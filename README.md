# Tesis_Final

Autor: Matias Gabriel Flores
Este repositorio posee los codigos realizados para el desarrollo practico de la Tesis de Grado de Ingeniería en Informatica en la Universidad Nacional de Avellaneda.

Se realizó la implementacion de distintos sistemas de recomendacion.
Para el desarrollo se utilizó la libreria Surprise de Nicolas Hug.
Las implementaciones de los sistemas se encuentran en la carpeta "Algoritmos".
Los modelos implementados son: Memory-Based  => User-Based
                                                Item-Based
                               Model-Based   => SVD_Funk
                                                PMF
                                                NMF
                               Content-Based => ContentKNN
                               Random        => Normal Predictor (Algoritmo aleatorio de la libreria surprise)


Los algoritmos a ejecutar son: KNN_User.py, KNN_Item.py, Random.py, SVD_Funk.py, PMF.py, NMF.py, Content_Based.py

Por defecto, al ejecutar: Se realizará la evaluacion de cada algoritmo con sus valores optimos de hiperparametro
                          Se realizará un top 10 de recomendaciones para el usuario 500

Como los calculos consumen mucha memoria, en la funcion evaluar se pasan por parametro dos booleanos: "ranking" y "caracteristicas"
Por defecto, evaluar() realiza la evaluacion de metrica de exactitud: RMSE, MSE, MAE y FCP.
"ranking" es para realizar mediciones en base a ranking de las predicciones: Precision, Recall y F1_Score. En caso de no querer realizarlo poner en "False".
"caracteristicas" es para realizar la medicion de otras medidas: Cobertura, Diversidad e Innovacion. En caso de no querer realizarlo poner en "False".
Se recomienda realizar la evaluacion de cada uno por separado en caso de que no se posea una PC con mas de 16gb de Ram.

CargadorDataset se encarga de cargar del path el dataset a utilizar y de ponerlo en formato para que lo sepa interpretar la libreria.
El dataset utilizado para las pruebas es el de ml_latest_small de grouplens.
Puede ser encontrado en su pagina en la seccion de datasets/movielens.
La carpeta "ml-latest-small" debe ser ubicada en la raiz de la carpeta del repositorio.

En Analisis Exploratorio de Datos se encuentra el archivo de jupyter notebook donde se pueden visualizar las implementaciones del analisis de los datasets junto a los graficos realizados.

En la carpeta Resultados se encuentran imagenes de los resultados obtenidos en la evaluacion de cada algoritmo.



El grid con todos los parametros se encuentra comentado, ya que consume mucho procesamiento.
Si se desea realizar la busqueda de hiperparametros, descomentar el grid del algoritmo, y comentar el que tiene los mejores.
