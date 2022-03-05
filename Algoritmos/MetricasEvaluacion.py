import itertools

from surprise import accuracy
from collections import defaultdict


class MetricasEvaluacion:

    def MAE(predicciones):
        return accuracy.mae(predicciones, verbose=False)

    def RMSE(predicciones):
        return accuracy.rmse(predicciones, verbose=False)

    def MSE(predicciones):
        return accuracy.mse(predicciones, verbose=False)

    def FCP(predicciones):
        return accuracy.fcp(predicciones, verbose=False)
    
    def Ranking(predicciones, k, limite):
        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predicciones:
            user_est_true[uid].append((est, true_r))
        
        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
        
            # Ordena los usuarios segun el valor estimado
            user_ratings.sort(key=lambda x: x[0], reverse=True)
        
            # Obtiene el numero de items relevantes
            n_rel = sum((true_r >= limite) for (_, true_r) in user_ratings)
        
            # Obtiene el numero de items recomendados en top k
            n_rec_k = sum((est >= limite) for (est, _) in user_ratings[:k])
        
            # Obtiene el numero de items relevantes y recomendados en top k
            n_rel_and_rec_k = sum(((true_r >= limite) and (est >= limite))
                                  for (est, true_r) in user_ratings[:k])
        
            # Precision@K: Proporcion de items recomendados que son relevantes
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        
            # Recall@K: Proporcion de items relevantes que son recomendados
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        
        return precisions, recalls
    
    def ResultadosRanking(predicciones, k, limite):
        precisions, recalls = MetricasEvaluacion.Ranking(predicciones, k, limite)
        
        promedioPrecision = sum(prec for prec in precisions.values()) / len(precisions)
        promedioRecall = sum(rec for rec in recalls.values()) / len(recalls)
        F1 = (2 * promedioPrecision * promedioRecall) / (promedioPrecision + promedioRecall)
        
        return promedioPrecision, promedioRecall, F1
    
    def ObtenerTopN(predicciones, n, rating_minimo):
        topN = defaultdict(list)

        for userID, movieID, actualRating, estimatedRating, _ in predicciones:
            if estimatedRating >= rating_minimo:
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    def Cobertura(topNPredichos, numero_total_usuarios, limite_rating):
        aciertos = 0
        for userID in topNPredichos.keys():
            acierto = False
            for movieID, predictedRating in topNPredichos[userID]:
                if predictedRating >= limite_rating:
                    acierto = True
                    break
            if acierto:
                aciertos += 1

        return aciertos / numero_total_usuarios

    def Diversidad(topNPredichos, algoritmoSimilitud):
        n = 0
        total = 0
        matrizSimilitud = algoritmoSimilitud.compute_similarities()
        for userID in topNPredichos.keys():
            pairs = itertools.combinations(topNPredichos[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = algoritmoSimilitud.trainset.to_inner_iid(str(movie1))
                innerID2 = algoritmoSimilitud.trainset.to_inner_iid(str(movie2))
                similarity = matrizSimilitud[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)

    def Innovacion(topNPredichos, rankings):
        n = 0
        total = 0
        for userID in topNPredichos.keys():
            for rating in topNPredichos[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n
