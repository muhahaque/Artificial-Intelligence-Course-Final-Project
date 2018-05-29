from sys import argv
import json
import numpy as np
import itertools
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation
from keras.utils import to_categorical

def main():
    """
    trainX: np array of training input in "onehot" format
    trainY: np array of cuisines for each recipe in training data
    testX: np array of testing input in "onehot" format
    test_id: np array of cuisines for recipes in test data
    """

    trainX, trainY, testX, test_id = read_file()
    results = {}

    results = randomForest(trainX, trainY, testX, test_id, results)
    results = neuralNet(trainX, trainY, testX, test_id, results)

    with open ("results.json", "w") as f:
        json.dump(results, f)

    # makegraphs(results)



def randomForest(trainX, trainY, testX, test_id, results):

    rf10 = RandomForestClassifier(n_estimators=10, verbose=1)
    rf10.fit(trainX, trainY)
    acc10 = rf10.score(testX, test_id)
    results["10_trees"] = {"accuracy":acc10}

    rf20 = RandomForestClassifier(n_estimators=20, verbose=1)
    rf20.fit(trainX, trainY)
    acc20 = rf20.score(testX, test_id)
    results["20_trees"] = {"accuracy":acc20}

    rf50 = RandomForestClassifier(n_estimators=50, verbose=1)
    rf50.fit(trainX, trainY)
    acc50 = rf50.score(testX, test_id)
    results["50_trees"] = {"accuracy":acc50}

    rf100 = RandomForestClassifier(n_estimators=100, verbose=1)
    rf100.fit(trainX, trainY)
    acc100 = rf100.score(testX, test_id)
    results["100_trees"] = {"accuracy":acc100}

    rf200 = RandomForestClassifier(n_estimators=200, verbose=1)
    rf200.fit(trainX, trainY)
    acc200 = rf200.score(testX, test_id)
    results["200_trees"] = {"accuracy":acc200}

    rf500 = RandomForestClassifier(n_estimators=500, verbose=1)
    rf500.fit(trainX, trainY)
    acc500 = rf500.score(testX, test_id)
    results["500_trees"] = {"accuracy":acc500}

    return results


def neuralNet(trainX, trainY, testX, test_id, results):

    label = LabelEncoder()
    trainY = label.fit_transform(trainY).astype(np.int32)
    label2 = LabelEncoder()
    testY = label2.fit_transform(test_id).astype(np.int32)
    trainY = to_categorical(trainY, 20)
    testY = to_categorical(testY, 20)


    epochs_model = Sequential()
    epochs_model.add(Dense(1024, activation='relu', input_shape=(trainX.shape[1],)))
    epochs_model.add(Dropout(0.4))
    epochs_model.add(Dense(512, activation='relu'))
    epochs_model.add(Dropout(0.4))
    epochs_model.add(Dense(256, activation='relu'))
    epochs_model.add(Dropout(0.4))
    epochs_model.add(Dense(128, activation='relu'))
    epochs_model.add(Dropout(0.4))
    epochs_model.add(Dense(20, activation='softmax'))
    epochs_model.summary()

    # train the model
    epochs_model.compile(optimizer="adagrad", loss="categorical_crossentropy", metrics=['accuracy'])
    history = epochs_model.fit(trainX, trainY, verbose=1, validation_data=(testX, testY), epochs=50)
    results["epochs"] = {k:v for k,v in history.history.items()}

    # loss, accuracy = neural_net.evaluate(testX, testY, verbose=0)
    # results["dense"] = {"loss":loss, "accuracy":accuracy}
    # print("accuracy: {}%".format(accuracy*100))

    return results


""" Reads data into training and test sets"""
def read_file():
    cuisines = {}

    with open ('train_copy_5.json') as train_f:
        train_data = json.loads(train_f.read())

    # CROSS VALIDATION #
    test_data = train_data[:8000]
    train_data = train_data[8000:]

    # Save train data to arrays
    train_X = np.array([x['ingredients'] for x in train_data])
    train_Y = np.array([y['cuisine'] for y in train_data])


    # list of all unique ingredients
    ingredient_list = list(set(itertools.chain(*train_X)))

    train_X_onehot = []
    test_X_onehot = []

    # OneHot representation of training data
    zlist = np.zeros(len(ingredient_list), dtype='bool')
    for recipe in train_X:
        onehot = zlist.copy()
        for i in range(len(ingredient_list)):
            if ingredient_list[i] in recipe:
                onehot[i] = 1
        train_X_onehot.append(onehot)

    train_X = np.array(train_X_onehot)

    # Save test data to arrays
    test_X = np.array([x['ingredients'] for x in test_data])
    test_id = np.array([e['cuisine'] for e in test_data])
    for recipe in test_X:
        onehot = zlist.copy()
        for i in range(len(ingredient_list)):
            if ingredient_list[i] in recipe:
                onehot[i] = 1
        test_X_onehot.append(onehot)
    test_X = np.array(test_X_onehot)

    return train_X, train_Y, test_X, test_id

def makegraphs(results):

    import matplotlib.pyplot as plt

    with open ("results.json") as f:
        results = json.load(f)

    ### Random Forest Bar Graph ###
    labels = ["10", "20", "50", "100", "200", "500"]
    accuracies = [results["10_trees"]["accuracy"], results["20_trees"]["accuracy"], results["50_trees"]["accuracy"], results["100_trees"]["accuracy"], results["200_trees"]["accuracy"], results["500_trees"]["accuracy"]]
    indices = np.arange(len(labels))
    plt.bar(indices, accuracies)
    plt.axis([-0.5,4.0,0.65,0.75])
    plt.xticks(indices, labels)
    plt.xlabel("number of trees")
    plt.ylabel("accuracy")
    plt.savefig("bar_graph.pdf")
    plt.clf()

    ### Neural Network Line Graph ###
    x = np.arange(1,51)
    y = results["epochs"]["acc"]
    plt.plot(x, y, label="training data")
    y = results["epochs"]["val_acc"]
    plt.plot(x, y, label="testing data")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("line_graph.pdf")


main()
