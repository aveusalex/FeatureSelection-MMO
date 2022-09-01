import numpy as np
from mealpy.evolutionary_based.GA import BaseGA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd


# a constante define o grau de importancia da quantidade de features na funcao fitness
lamda = 3
qtd_features_maximas = 4000

# vamos determinar que um array binário represente as features de nosso dataset,
# sendo 1 quando a feature estará presente e 0 quando ausente.
# Isso elimina o problema de reptição de features, ao mesmo tempo que é simples.
# df_train = pd.read_csv('../Dados/train.csv')
# df_test = pd.read_csv('../Dados/test.csv')



# após o pré-processamento, essas devem ser as variaveis que recebem os dados
X_train_final = None
y_train_final = None

X_val_final = None
y_val_final = None

colunas = df_train.drop(columns=['beer_id', 'target']).columns

def classificador(data_to_train, data_to_test, y_train, y_test):
    # instanciando o classificador
    clf = LogisticRegression(solver='liblinear')
    clf.fit(data_to_train, y_train)
    y_chapeu = clf.predict(data_to_test)

    return accuracy_score(y_test, y_chapeu)


def fitness(features):
    qtd_features = features.sum()  # as features entram como vetor binario, por isso .sum()

    # se a quantidade de features for maior que a quantidade máxima, retorna um valor altíssimo
    if qtd_features > qtd_features_maximas:
        return 999999

    # transformando o vetor binario para booleano
    features = np.where(features == 1, True, False)

    data_to_train = X_train_final.loc[:, features]
    data_to_test = X_val_final.loc[:, features]
    acuracia = classificador(data_to_train, data_to_test, y_train_final, y_val_final)

    # unindo os dois objetivos em uma unica expressao, tornar um problema de minimizacao
    return (1/acuracia) + lamda * qtd_features/qtd_features_maximas


problem_dict = {
    'fitness': fitness,
    'lb': np.zeros(len(colunas)),
    'ub': np.ones(len(colunas)),
    'minmax': 'min',
}

epoch = 1000
pop_size = 100
pc = 0.9
pm = 0.05
selection = 'roulette'
model1 = BaseGA(problem_dict, epoch, pop_size, pc, pm, crossover='multi_points', selection=selection, n_jobs=-1)
best_position, best_fitness = model1.solve()
print(f"Solution: {best_position}, Fitness: {best_fitness}")
