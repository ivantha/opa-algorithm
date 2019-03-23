import numpy as np
import csv


def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generateReport(data, y, pred, pa, n, type):
    file_name = 'pa {} - iter {} - {}'.format(pa, n, type)
    with open('./output/{}.csv'.format(file_name), mode='w') as outputFile:
        writer = csv.writer(outputFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['sample_code_number', 'class', 'PREDICTION'])

        for i in range(len(y)):
            writer.writerow([data[i, 0], y[i], pred[i]])


def predict(x, w):
    y = []
    for i in range(len(x)):
        y.append(sign(np.dot(w, x[i])))
    return y


def getAccuracy(y, pred):
    trues = 0
    for i in range(len(y)):
        if y[i] == pred[i]:
            trues += 1
    return trues / len(y) * 100


if __name__ == '__main__':
    data = np.genfromtxt('../../data/datafile.csv', delimiter=',')

    # class : 4 -> 1, 2 -> -1
    for i in range(len(data)):
        if data[i, 10] == 2:
            data[i, 10] = -1
        else:
            data[i, 10] = 1

    X = data[:, 1:10]
    Y = data[:, 10]

    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    for i in range(len(data)):
        if i <= int(len(data) * 2 / 3):
            xTrain.append(X[i])
            yTrain.append(Y[i])
        else:
            xTest.append(X[i])
            yTest.append(Y[i])

    n = int(input('Please enter the desired number of iterations (>0) : '))
    pa = int(input('Please enter the version of the PA algorithm (0,1,2) : '))
    c = 1

    # 9 features
    w = np.zeros(9)

    trainAccList = []
    testAccList = []

    for iter in range(n):
        # train
        for i in range(len(xTrain)):
            x = xTrain[i]
            y = yTrain[i]

            l = max(0, 1 - (y * np.dot(w, x)))

            k = None
            if pa == 0:
                k = l / np.power(np.linalg.norm(x), 2)
            elif pa == 1:
                k = min(c, l / np.power(np.linalg.norm(x), 2))
            elif pa == 2:
                k = l / (np.power(np.linalg.norm(x), 2) + (1 / (2 * c)))

            w = w + (k * y * x)

        yTrainPred = predict(xTrain, w)
        yTestPred = predict(xTest, w)

        generateReport(data, yTrain, yTrainPred, pa, iter, 'train')
        generateReport(data, yTest, yTestPred, pa, iter, 'test')

        trainAccList.append(getAccuracy(yTrainPred, yTrain))
        testAccList.append(getAccuracy(yTestPred, yTest))

    print('Training accuracy')
    for i in range(n):
        print(trainAccList[i])

    print('---------------------------------------------------------')

    print('Testing accuracy')
    for i in range(n):
        print(testAccList[i])
