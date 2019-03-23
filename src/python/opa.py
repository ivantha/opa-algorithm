import numpy as np
import pandas as pd


def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def preprocess_dataset():
    # load data
    data = pd.read_csv('../../data/datafile.csv', delimiter=",")

    # assign column names > for easiness in referencing
    data.columns = [
        'sample_code_number',
        'clump_thickness',
        'uniformity_of_cell_size',
        'uniformity_of_cell_shape',
        'marginal_adhesion',
        'single_epithelial_cell_size',
        'bare_nuclei',
        'bland_chromatin',
        'normal_nucleoli',
        'mitoses',
        'class'
    ]

    # replace class (2, 4) with (-1, +1)
    data['class'] = data['class'].map({4: 1, 2: -1})

    df_X = data.loc[:, 'clump_thickness':'mitoses']
    df_Y = data['class']

    # mask to split data into train and test sets randomly
    # msk = np.random.rand(len(data)) < (2 / 3)

    q = int(len(data) * 2 / 3)
    msk = ([True] * q) + ([False] * (len(data) - q))
    msk = np.asarray(msk)

    # data, x_train, y_train, x_test, y_test
    return data, df_X[msk], df_Y[msk], df_X[~msk], df_Y[~msk]


# generate online passive aggressive function
def generate_opa(pa, c):
    if pa == 0:
        def lagrange_multiplier(loss, x):
            return loss / np.power(np.linalg.norm(x), 2)
    elif pa == 1:
        def lagrange_multiplier(loss, x):
            return min(c, loss / np.power(np.linalg.norm(x), 2))
    elif pa == 2:
        def lagrange_multiplier(loss, x):
            return loss / (np.power(np.linalg.norm(x), 2) + (1 / (2 * c)))

    def opa(x, y, w):
        loss = max(0, 1 - (y * np.dot(w, x)))
        w_new = w + (lagrange_multiplier(loss, x) * y * x)
        return w_new, loss

    return opa


def train(df, df_x_train, df_y_train, df_x_test, df_y_test, n_iter=1, pa=1, c=1):
    num_features = len(df_x_train.columns)  # number of features
    w = np.zeros(num_features)  # weights

    opa = generate_opa(pa, c)

    train_accuracies = []
    test_accuracies = []

    for i in range(n_iter):
        # train
        for index, row in df_x_train.iterrows():
            x = row.to_numpy()
            y = df_y_train[index]
            w, loss = opa(x, y, w)

        # get predictions
        y_pred_train = get_prediction(df_x_train, w)
        y_pred_test = get_prediction(df_x_test, w)

        # save output to a csv
        save_output(df, df_y_train, y_pred_train, pa, i, 'train')
        save_output(df, df_y_test, y_pred_test, pa, i, 'test')

        # calculate accuracies
        train_accuracy = (y_pred_train == df_y_train).mean() * 100
        test_accuracy = (y_pred_test == df_y_test).mean() * 100

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    return train_accuracies, test_accuracies


def get_prediction(df_x, w):
    y_pred = pd.Series()

    for index, row in df_x.iterrows():
        x = row.to_numpy()
        y_pred.loc[index] = sign(np.dot(w, x))

    return y_pred


def save_output(df, df_y, y_pred, pa, n_iter, type):
    pd_out = pd.concat(
        [
            df['sample_code_number'],
            df_y,
            y_pred
        ],
        axis=1,
        sort=False,
        join='inner'
    )
    pd_out.columns = ['sample_code_number', 'class', 'PREDICTION']

    file_name = 'pa {} - iter {} - {}'.format(pa, n_iter, type)
    pd_out.to_csv('./output/{}.csv'.format(file_name), encoding='utf-8', sep=',')


def print_train_results(pa, n_iter, train_result):
    print('PA version :', pa)
    print('Iteration count :', n_iter)
    print('')

    print('Accuracy of training set over the iterations -> ')
    for i in range(n_iter):
        print('{} : {}'.format(i + 1, train_result[0][i]))
    print('')

    print('Accuracy of testing set over the iterations -> ')
    for i in range(n_iter):
        print('{} : {}'.format(i + 1, train_result[1][i]))
    print('')


if __name__ == '__main__':
    df, df_x_train, df_y_train, df_x_test, df_y_test = preprocess_dataset()

    n_iter = int(input('Please enter the desired number of iterations (>0) : '))
    pa = int(input('Please enter the version of the PA algorithm (0,1,2) : '))
    c = int(input('Please enter the desired c value (>0) : '))

    print_train_results(pa, n_iter, train(df, df_x_train, df_y_train, df_x_test, df_y_test, n_iter=n_iter, pa=pa, c=c))
