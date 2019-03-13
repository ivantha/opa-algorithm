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

    return df_X[msk], df_X[~msk], df_Y[msk], df_Y[~msk]


def generate_opa(c):
    def opa(x, y, w):
        loss = max(0, 1 - (y * np.dot(w, x)))
        lagrange_multiplier = min(c, loss / np.power(np.linalg.norm(x), 2))
        w_new = w + (lagrange_multiplier * y * x)

        return w_new, loss

    return opa

def get_accuracy(x_values, y_values, w):



if __name__ == '__main__':
    df_x_train, df_x_test, df_y_train, df_y_test = preprocess_dataset()

    num_features = len(df_x_train.columns)  # number of features
    w = np.zeros(num_features)  # weights

    c = 1
    opa = generate_opa(c)  # generate online passive aggressive function

    for i in range(10):
        for index, row in df_x_train.iterrows():
            x = row.to_numpy()
            y = df_y_train[index]
            y_pred = sign(np.dot(w, x))

            w, loss = opa(x, y, w)

        total = 0
        count = 0
        for index, row in df_x_test.iterrows():
            x = row.to_numpy()
            y = df_y_test[index]
            y_pred = sign(np.dot(w, x))

            accuracy = (1 - abs((y_pred - y) / y)) * 100

            total += accuracy
            count += 1

        print(total/count)
