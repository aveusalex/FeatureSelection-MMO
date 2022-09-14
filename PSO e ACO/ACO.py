import numpy as np
from mealpy.swarm_based.ACOR import BaseACOR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def main(qtd_features_maximas):
    clf = RandomForestClassifier(criterion='entropy', n_estimators=100)
    def classificador(data_to_train, data_to_test, y_train, y_test):
        # instanciando o classificador
        clf.fit(data_to_train, y_train)
        y_chapeu = clf.predict(data_to_test)

        return accuracy_score(y_test, y_chapeu)

    def fitness(features):
        # agora a quantidade de features sempre será fixa
        # precisamos lidar com valores repetidos

        if features.shape[0] - np.unique(features).shape[0] != 0:
            return 10000
        # precisamos de valores inteiros pois são sempre os indices das features a considerar
        features = np.round(features).astype(np.int32).astype(str)

        data_to_train = X_train_final.loc[:, features]
        data_to_test = X_val_final.loc[:, features]

        acuracia = classificador(data_to_train, data_to_test, y_train_final, y_val_final)

        # unindo os dois objetivos em uma unica expressao, tornar um problema de minimizacao
        return 1/acuracia

    problem_dict = {
        'fit_func': fitness,
        'lb': 0,
        'ub': 194999,
        'minmax': 'min',
        'save_population': False,
        'n_dims': qtd_features_maximas,
    }

    epoch = 10000
    pop_size = 100
    sample_count = 50
    intent_factor = 0.5
    zeta = 1.0

    model1 = BaseACOR(problem_dict, epoch, pop_size, sample_count, intent_factor, zeta)
    best_position, best_fitness = model1.solve()
    print(f"SolutionACO {qtd_features_maximas}: {best_position}, Fitness: {best_fitness}")

    ## Salvando as features encontradas em um arquivo txt
    with open(f"../Features/ACO_features_{qtd_features_maximas}_{best_fitness:.2f}.txt", "w") as f:
        for i in best_position:
            f.write(str(i) + '\n')

    # a partir desse ponto gera-se os dados de teste para submissao no kaggle
    best_position.sort()
    features_finais = np.round(best_position).astype(np.int32).astype(str)
    val = pd.read_csv("../Dados/test.csv", dtype="float64")
    beer_id = val.beer_id.astype(np.int32)
    data_to_test = val.loc[:, features_finais]
    data_to_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_to_test.fillna(data_to_test.mean(), inplace=True)

    # classificando os dados de teste
    # treinando nos dados
    clf = RandomForestClassifier(criterion='entropy', n_estimators=100)
    clf.fit(X_train_final.loc[:, features_finais], y_train_final)
    predictions = clf.predict(data_to_test.astype("float32"))
    submission = pd.DataFrame(np.stack((beer_id, predictions), axis=1),
                              columns=["beer_id", "target"]).astype(dtype={"beer_id":"int32", "target":"float64"})
    submission.to_csv(f"../Submissions/SubmissionACO{qtd_features_maximas}.csv", index=False)


if __name__ == '__main__':
    import threading
    import logging

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    # após o pré-processamento, essas devem ser as variaveis que recebem os dados
    X_train_final = pd.read_csv('../Dados/DataTrain.csv', dtype="float32")
    y_train_final = X_train_final['target']

    X_val_final = pd.read_csv('../Dados/DataVal.csv', dtype="float32")
    y_val_final = X_val_final['target']

    colunas = X_train_final.columns

    qtd_features = [150, 200, 250, 300]

    threads = list()
    for index in range(len(qtd_features)):
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=main, args=(qtd_features[index],))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)
