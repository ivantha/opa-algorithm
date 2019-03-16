import numpy as np
import pandas as pd


def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def preprocess_dataset():
    # load data
    data = pd.read_csv("../../data/datafile.csv", delimiter=",")

    # assign column names > for easiness in referencing
    data.columns = [
        "sample_code_number",
        "clump_thickness",
        "uniformity_of_cell_size",
        "uniformity_of_cell_shape",
        "marginal_adhesion",
        "single_epithelial_cell_size",
        "bare_nuclei",
        "bland_chromatin",
        "normal_nucleoli",
        "mitoses",
        "class"
    ]

    # remove 1st attribute
    data = data.drop([data.columns[0]], axis=1)

    # replace class (2, 4) with (-1, +1)
    data['class'] = data['class'].map({4: 1, 2: -1})

    df_X = data.loc[:, "clump_thickness":"mitoses"]
    df_Y = data["class"]

    # mask to split data into train and test sets
    msk = np.random.rand(len(data)) < (2 / 3)

    msk = []
    for i in range(len(data)):
        if i <= (int(len(data) * 2 / 3)):
            msk += [True]
        else:
            msk += [False]

    msk = np.asarray(msk)

    # x_train, y_train, x_test, y_test
    return df_X[msk], df_Y[msk], df_X[~msk], df_Y[~msk]


def generate_opa(c, pa=1):
    if pa == 0:
        def opa(x, y, w):
            loss = max(0, 1 - (y * np.dot(w, x)))
            lagrange_multiplier = loss / np.power(np.linalg.norm(x), 2)
            w_new = w + (lagrange_multiplier * y * x)
            return w_new, loss

        return opa

    elif pa == 1:
        def opa(x, y, w):
            loss = max(0, 1 - (y * np.dot(w, x)))
            lagrange_multiplier = min(c, loss / np.power(np.linalg.norm(x), 2))
            w_new = w + (lagrange_multiplier * y * x)
            return w_new, loss

        return opa

    elif pa == 2:
        def opa(x, y, w):
            loss = max(0, 1 - (y * np.dot(w, x)))
            lagrange_multiplier = loss / (np.power(np.linalg.norm(x), 2) + (1 / (2 * c)))
            w_new = w + (lagrange_multiplier * y * x)
            return w_new, loss

        return opa

    else:
        return -1


def train(df_x_train, df_y_train, df_x_test, df_y_test, n_iter=1, pa=1):
    num_features = len(df_x_train.columns)  # number of features
    w = np.zeros(num_features)  # weights

    c = 1
    opa = generate_opa(c, pa)  # generate online passive aggressive function

    train_accuracies = []
    test_accuracies = []

    for i in range(n_iter):
        # train
        for index, row in df_x_train.iterrows():
            x = row.to_numpy()
            y = df_y_train[index]
            w, loss = opa(x, y, w)

        # get accuracy
        train_accuracies.append(get_accuracy(df_x_train, df_y_train, w))
        test_accuracies.append(get_accuracy(df_x_test, df_y_test, w))

    return train_accuracies, test_accuracies


def get_accuracy(df_x, df_y, w):
    y_pred = []

    for index, row in df_x.iterrows():
        x = row.to_numpy()
        y_pred.append(sign(np.dot(w, x)))

    accuracy = ((1 - abs((np.asarray(y_pred) - df_y) / df_y)) * 100).mean()
    return accuracy


def print_train_results(pa, train_result):
    print(pa + "_______________________________________")
    print(train_result[0])
    print(np.asarray(train_result[0]).mean())
    print(train_result[1])
    print(np.asarray(train_result[1]).mean())
    print("")


if __name__ == '__main__':
    df_x_train, df_y_train, df_x_test, df_y_test = preprocess_dataset()

    print_train_results("pa0", train(df_x_train, df_y_train, df_x_test, df_y_test, n_iter=10, pa=0))
    print_train_results("pa1", train(df_x_train, df_y_train, df_x_test, df_y_test, n_iter=10, pa=1))
    print_train_results("pa2", train(df_x_train, df_y_train, df_x_test, df_y_test, n_iter=10, pa=2))
