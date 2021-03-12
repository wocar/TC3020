#!/usr/bin/env python
import math

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from william import load_default_dataset, WilliamKNN, sci_to_pd, load_gender_dataset, WilliamLogisticRegression
import pandas as pd

ks_to_test = [1, 2, 3, 5, 10, 15, 20, 25, 50, 100]
#ks_to_test = [1, 2, 20 ] # Debug
ps_to_tests = [1, 2, math.inf]
datasets_to_test = [{'dataset': sci_to_pd(load_iris()), 'target_column': 'target', 'name': "Iris"},

                    {'dataset': load_default_dataset(as_frame=True)   #.sample(150, random_state=1) # Debug, tomamos menos samples para que sea mas rapido
                        ,
                     'target_column': 'default',
                     'name': "Default"},

                    {'dataset': load_gender_dataset(as_frame=True)   #.sample(150, random_state=1)
                        ,
                     'target_column': 'Gender',
                     'name': "Genero"}
                    ]


def punto_6():
    results = []
    for dataset in datasets_to_test:
        df = dataset['dataset']
        X = df.drop(dataset["target_column"], axis=1)
        y = df[dataset["target_column"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                            random_state=1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = WilliamKNN()

        for k in ks_to_test:
            for p in ps_to_tests:
                print(f"Training: '{dataset['name']}' p={p} k={k}")

                y_hat = model.knn_predict(X_train, X_test, y_train, k=k, p=p)
                accuracy = accuracy_score(y_test, y_hat)
                print(f"Accuracy: {accuracy}")
                results.append([dataset['name'], k, p, accuracy, y_hat])

    df = pd.DataFrame(results, columns=['dataset', 'k', 'p', 'accuracy', 'y_hat'])

    best_combinations = []
    plt.style.use('seaborn-whitegrid')
    fig, axs = plt.subplots(3, 3)


    fila = 0
    columna = 0
    for _, datasetGroup in df.groupby(['dataset']):
        dsName = datasetGroup.dataset.values[0]

        for _, pGroup in datasetGroup.groupby(['p']):
            pvalue = str(pGroup.p.values[0])
            axs[fila, columna].set_title(dsName + ". P = " + pvalue)
            axs[fila, columna].scatter(pGroup.k, pGroup.accuracy)
            axs[fila, columna].set_xlabel("k")
            axs[fila, columna].set_ylabel("Accuracy")
            columna += 1

        fila += 1
        columna = 0
        better_result = datasetGroup.loc[datasetGroup['accuracy'].idxmax()]
        best_combinations.append([dsName, better_result.k, better_result.p, better_result.accuracy, better_result.y_hat])

    fig.tight_layout()
    plt.show()
    return pd.DataFrame(best_combinations, columns=['name', 'k', 'p', 'accuracy', 'y_hat']), df


# Hasta aquí ya  terminamos hasta el punto 6 de la parte 1
# "Por cada dataset identifica que combinacion de funcíon de distancia con valor de k produjo el mejor valor de precision."

best, results = punto_6()

# Parte 2

for index, row in best.iterrows():
    ds = row['name']
    k = row['k']
    dataset = list(filter(lambda d: d['name'] == ds, datasets_to_test))[0]

    df = dataset['dataset']
    X = df.drop(dataset["target_column"], axis=1)
    y = df[dataset["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    model.fit(X_train, y_train)
    y_hat_test_sci_knn = model.predict(X_test)
    sci_knn_accuracy = accuracy_score(y_test, y_hat_test_sci_knn)

    model = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    model.fit(X_train, y_train)
    y_hat_test_sci_knn_peor = model.predict(X_test)
    sci_knn_accuracy_peor = accuracy_score(y_test, y_hat_test_sci_knn_peor)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_hat_test_sci_decision = model.predict(X_test)
    sci_decision_accuracy = accuracy_score(y_test, y_hat_test_sci_decision)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_hat_test_sci_logistic = model.predict(X_test)
    sci_logistic_accuracy = accuracy_score(y_test, y_hat_test_sci_logistic)

    model = WilliamLogisticRegression(iterations=3000)
    model.fit(X_train, y_train)
    y_hat_test_william_logistic = model.predict(X_test)
    william_logistic_accuracy = accuracy_score(y_test, y_hat_test_william_logistic)

    print(f"{ds}")
    print(f"*" * 50)

    print(f"Scikit KNN (k={k}): {sci_knn_accuracy}")
    print(f"William KNN best (k={k},p={row['p']}): {row['accuracy']}")
    print(f"Decision Tree: {sci_decision_accuracy}")
    print(f"Logistic Regression: {sci_logistic_accuracy}")
    print(f"William Logistic Regression: {sci_logistic_accuracy}")

    if ds == 'Genero':
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.5, cmap='seismic_r')
        axs[0, 0].set_title("Clasificacion Real")
        axs[0, 0].set_xlabel('Height')
        axs[0, 0].set_ylabel('Weight')

        axs[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_hat_test_sci_decision, alpha=0.5, cmap='seismic_r')
        axs[0, 1].set_title("Arbol de decision")
        axs[0, 1].set_xlabel('Height')
        axs[0, 1].set_ylabel('Weight')

        axs[0, 2].scatter(X_test[:, 0], X_test[:, 1], c=y_hat_test_sci_logistic, alpha=0.5, cmap='seismic_r')
        axs[0, 2].set_title("Regresion lineal")
        axs[0, 2].set_xlabel('Height')
        axs[0, 2].set_ylabel('Weight')

        axs[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=row['y_hat'], alpha=0.5, cmap='seismic_r')
        axs[1, 0].set_title(f"Best-will KNN\np={row['p']} k={row['k']} acc={row['accuracy']}")
        axs[1, 0].set_xlabel('Height')
        axs[1, 0].set_ylabel('Weight')

        y_hat_test_sci_knn_peor = results[results["dataset"] == "Genero"]
        y_hat_test_sci_knn_peor = y_hat_test_sci_knn_peor[y_hat_test_sci_knn_peor["k"] == 20]
        y_hat_test_sci_knn_peor = y_hat_test_sci_knn_peor[y_hat_test_sci_knn_peor["p"] == math.inf].iloc[0].y_hat

        axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_hat_test_sci_knn_peor, alpha=0.5, cmap='seismic_r')
        axs[1, 1].set_title(f"Worst KNN\nk={1} acc={accuracy_score(y_hat_test_sci_knn_peor, y_test)}")
        axs[1, 1].set_xlabel('Height')
        axs[1, 1].set_ylabel('Weight')
        axs[1, 1].set_ylabel('Weight')


        axs[1, 2].scatter(X_test[:, 0], X_test[:, 1], c=y_hat_test_sci_knn, alpha=0.5, cmap='seismic_r')
        axs[1, 2].set_title(f"SciKi K-NN\nk={row['k']}, acc={sci_knn_accuracy}")
        axs[1, 2].set_xlabel('Height')
        axs[1, 2].set_ylabel('Weight')


        fig.tight_layout()
        plt.show()




