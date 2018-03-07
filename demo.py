import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
import keras.optimizers
from sklearn.utils import shuffle
from sklearn.svm import SVC
import seaborn as sb

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 0)

train_df = pd.read_csv('data/train.csv')
train_df = shuffle(train_df)
test_df = pd.read_csv('data/test.csv')
test_df = shuffle(test_df)

print(train_df.describe())

def create_plot(subset=train_df, x='Age', y='Fare'):
    markers = {0: 'x', 1: 'o'}
    colors = {'female': 'red', 'male': 'blue'}
    labels = {(0, 'female'): 'female deceased', (1, 'female'): 'female survived',
              (0, 'male'): 'male deceased', (1, 'male'): 'male survived'}

    fig, ax = plt.subplots()
    # ax.set_ylim([-3, 100])
    grouped = subset.groupby(['Survived', 'Sex'])

    for key, group in grouped:
        group.plot(ax=ax, x=x, y=y, kind='scatter', marker=markers[key[0]], color=colors[key[1]], label=labels[key])

def create_heatmap(df):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sb.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.show()

no_fare = train_df.query('Fare != Fare')
print('no fare: {}'.format(len(no_fare)))
# create_plot(babies)
# plt.show()

def show_survival_rate(subset=train_df, desc=''):
    survived = len(subset[subset['Survived'] == 1])
    total = len(subset)
    print('{}: survival rate {:.1f}% ({}/{})'.format(desc, survived/total*100, survived, total))


def first_char(x):
    if not isinstance(x, str):
        return None
    return x[0]


def starts_with_letter(x):
    return first_char(x).isalpha()


# show_survival_rate(df, 'Total')
# for key, group in df.groupby('Sex'):
#     show_survival_rate(group, key)

def prepare_data(df):
    df['Age'] = np.where((np.isnan(df['Age'])) &
                         ((df['Name'].str.contains('Miss')) | (df['Name'].str.contains('Mlle')) |
                          (df['Name'].str.contains('Ms.')) | (df['Name'].str.contains('Master'))), 8, df['Age'])

    # df['Age'] = np.where((np.isnan(df['Age'])) & (df['Name'].str.contains('Master')), 8, df['Age'])
    df['Age'] = np.where(np.isnan(df['Age']), 30, df['Age'])

    # standardize age
    # max_age = df.Age.max()
    # df.Age = df.Age / max_age

    df['male'] = np.where(df['Sex'] == 'male', 1, 0)

    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
    df = pd.concat([df, embarked_dummies], axis=1)

    # standardize fare
    df['Fare'] = np.where(np.isnan(df['Fare']), 0, df['Fare'])
    max_fare = df.Fare.max()
    df.Fare = df.Fare / max_fare

    df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Pclass'], axis=1)
    return df


train_df = prepare_data(train_df)
print(train_df.head(20))
create_heatmap(train_df)
# create_plot()
# plt.show()

target_col = 'Survived'
x_train = train_df.drop(target_col, axis=1).values
y_train = train_df[target_col].values


# define neural network
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # model.summary()

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, validation_split=None, validation_data=None):
    if validation_split or validation_data:
        weights_file = 'output/weights_best_model.hdf5'
        callbacks = [ModelCheckpoint(filepath=weights_file, verbose=0, save_best_only=True)]
    else:
        callbacks = None

    hist = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
                     validation_split=validation_split, validation_data=validation_data, verbose=0)

    return hist
    # if validation_split:
    #     model.load_weights(weights_file)


def evaluate_model(model, x_val, y_val):
    score = model.evaluate(x_val, y_val, verbose=0)
    return score


def predict(model, x_test):
    prediction = model.predict(x_test)
    return (prediction >= 0.5).astype(int)


def run_cross_validation(x, y):
    # seed = 7
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    scores = []

    for train, val in kfold.split(x, y):
        model = create_model(x.shape[1])
        train_model(model, x[train], y[train], validation_data=(x[val], y[val]))
        score = evaluate_model(model, x[val], y[val])
        print('score: {}'.format(score[1]))
        scores.append(score[1])

    print('score avg: {}, score stdv: {}'.format(np.mean(scores), np.std(scores)))


def create_result(x_train, y_train, df):
    final_model = create_model(x_train.shape[1])
    train_model(final_model, x_train, y_train, validation_split=0.2)
    final_model.load_weights('output/weights_best_model.hdf5')

    test_passenger_ids = df.PassengerId.values
    test_passenger_ids.shape = (test_passenger_ids.shape[0], 1)
    df = prepare_data(df)
    # if 'Cabin_T' not in test_df:
    #     test_df['Cabin_T'] = 0
    x_test = df.values

    prediction = predict(final_model, x_test)
    result = np.concatenate((test_passenger_ids, prediction), axis=1)
    np.savetxt("output/result.csv", result, delimiter=",", fmt='%1i', header='PassengerId,Survived', comments='')
    return final_model


def visualize_model(model, history):
    plt.figure(1)
    h = history.history
    print('acc: {}, val acc: {}, loss: {}, val loss: {}'.format(h['acc'][-1], h['val_acc'][-1], h['loss'][-1], h['val_loss'][-1]))
    # summarize history for accuracy
    plt.subplot(221)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    # summarize history for loss
    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


# clf = SVC(probability=True)
# clf.fit(x_train, y_train)
#
# test_df_prepared = prepare_data(test_df)
# x_test = test_df_prepared.values
#
# prediction = clf.predict(x_test)
# print(prediction)
#
# test_passenger_ids = test_df.PassengerId.values
# test_passenger_ids.shape = (test_passenger_ids.shape[0], 1)
# prediction.shape = (prediction.shape[0], 1)
# result = np.concatenate((test_passenger_ids, prediction), axis=1)
# np.savetxt("output/result.csv", result, delimiter=",", fmt='%1i', header='PassengerId,Survived', comments='')



# run_cross_validation(x_train, y_train)

# model = create_result(x_train, y_train, test_df)
#
# model = create_model(x_train.shape[1])
# hist = train_model(model, x_train, y_train, validation_split=0.2)
# visualize_model(model, hist)



