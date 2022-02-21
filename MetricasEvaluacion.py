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
    
    def Ranking(predicciones, k, threshold):
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
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        
            # Obtiene el numero de items recomendados en top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        
            # Obtiene el numero de items relevantes y recomendados en top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])
        
            # Precision@K: Proporcion de items recomendados que son relevantes

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        
            # Recall@K: Proporcion de items relevantes que son recomendados
            # When n_rel is 0, Recall is undefined. We here set it to 0.
        
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        
        return precisions, recalls
    
    def ResultadosRanking(predicciones, k, threshold):
        precisions, recalls = MetricasEvaluacion.Ranking(predicciones, k, threshold)
        
        PromedioPrecision = sum(prec for prec in precisions.values()) / len(precisions)
        PromedioRecall = sum(rec for rec in recalls.values()) / len(recalls)
        F1 = (2 * PromedioPrecision * PromedioRecall) / (PromedioPrecision + PromedioRecall)
        
        return PromedioPrecision, PromedioRecall, F1
    
    def ObtenerTopN(predicciones, n=10, minimumRating=4.0):
        topN = defaultdict(list)

        for userID, movieID, actualRating, estimatedRating, _ in predicciones:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    # What percentage of users have at least one "good" recommendation
    def Cobertura(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    def Diversidad(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)

    def Innovacion(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n
