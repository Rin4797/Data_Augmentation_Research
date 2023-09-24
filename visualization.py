import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle

import tensorflow as tf

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import multilabel_confusion_matrix


def draw_dataframe(df, title):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.title(title, fontsize=18)
    fig.tight_layout()
    plt.show()


class DataVisualization:

    def __init__(self, label_names, x_train, y_train, x_test, y_test, x_val=np.zeros((0, 3072)), y_val=np.zeros((0, )),
                 y_vect_train=np.zeros((0, 10)), y_vect_test=np.zeros((0, 10)), y_vect_val=np.zeros((0, 10)),
                 x_image_train=np.zeros((0, 32, 32, 3)), x_image_test=np.zeros((0, 32, 32, 3)),
                 x_image_val=np.zeros((0, 32, 32, 3))):
        self.label_names = label_names
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val

        self.image_height = self.x_train.shape[1]
        self.image_width = self.x_train.shape[2]

        self.y_vect_train = y_vect_train
        self.y_vect_test = y_vect_test
        self.y_vect_val = y_vect_val
        self.x_image_train = x_image_train
        self.x_image_test = x_image_test
        self.x_image_val = x_image_val

        self.batches_analysis = None

    def __add__(self, other):
        return DataVisualization(self.label_names, np.concatenate((self.x_train, other.x_train), axis=0),
                                 np.concatenate((self.y_train, other.y_train), axis=0),
                                 np.concatenate((self.x_test, other.x_test), axis=0),
                                 np.concatenate((self.y_test, other.y_test), axis=0),
                                 np.concatenate((self.x_val, other.x_val), axis=0),
                                 np.concatenate((self.y_val, other.y_val), axis=0),
                                 np.concatenate((self.y_vect_train, other.y_vect_train), axis=0),
                                 np.concatenate((self.y_vect_test, other.y_vect_test), axis=0),
                                 np.concatenate((self.y_vect_val, other.y_vect_val), axis=0),
                                 np.concatenate((self.x_image_train, other.x_image_train), axis=0),
                                 np.concatenate((self.x_image_test, other.x_image_test), axis=0),
                                 np.concatenate((self.x_image_val, other.x_image_val), axis=0),)


    def shuffle_data(self):
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        self.x_test, self.y_test = shuffle(self.x_test, self.y_test)

    def make_val(self, val_size=None):
        if val_size is None:
            val_size = int(self.x_train.shape[0] * 0.2)
        self.x_val = self.x_train[-val_size:]
        self.y_val = self.y_train[-val_size:]
        self.x_train = self.x_train[:-val_size]
        self.y_train = self.y_train[:-val_size]

    def make_data(self):
        def image_format(image_data):
            return np.transpose(image_data.reshape(self.image_height, self.image_width, 3, order='F'), axes=[1, 0, 2])

        self.y_vect_train = tf.keras.utils.to_categorical(self.y_train)
        self.y_vect_test = tf.keras.utils.to_categorical(self.y_test)
        self.y_vect_val = tf.keras.utils.to_categorical(self.y_val)

        self.x_image_train = np.array([image_format(i) for i in self.x_train])
        self.x_image_test = np.array([image_format(i) for i in self.x_test])
        self.x_image_val = np.array([image_format(i) for i in self.x_val])

    def make_data_stat(self):
        def data_analysis(labels):
            unique, counts = np.unique(labels, return_counts=True)
            unique_names = ['' for i in range(len(unique))]
            for i in range(len(unique)):
                unique_names[i] = self.label_names[np.int64(unique[i])]
            return dict(zip(unique_names, counts))

        self.batches_analysis = pd.DataFrame(columns=self.label_names)
        rows = [data_analysis(self.y_train), data_analysis(self.y_val), data_analysis(self.y_test)]
        self.batches_analysis = pd.DataFrame()
        self.batches_analysis['name'] = ['train', 'val', 'test']
        self.batches_analysis['size'] = [self.y_train.shape[0], self.y_val.shape[0], self.y_test.shape[0]]
        self.batches_analysis = self.batches_analysis.join(pd.DataFrame.from_dict(rows, orient='columns'))
        # self.batches_analysis['average'] = np.average(self.batches_analysis[self.label_names], axis=1)
        self.batches_analysis['std'] = np.std(self.batches_analysis[self.label_names], axis=1)

    def size_data_analysis(self):
        return self.batches_analysis[['name', 'size', 'std']]

    def class_data_analysis(self):
        return self.batches_analysis[['name'] + self.label_names]


class ModelVisualization:

    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.history = None
        self.predictions = None
        self.confusion_matrix = None

    def model_fit(self, epochs):
        self.history = self.model.fit(self.data.x_image_train, self.data.y_vect_train, batch_size=64, epochs=epochs,
                                      validation_data=(self.data.x_image_val, self.data.y_vect_val))

    def get_history(self):
        return self.history.history

    def draw_model_plot(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epoch_range = range(len(acc))

        plt.plot(epoch_range, acc, label='Training accuracy')
        plt.plot(epoch_range, val_acc, label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epoch_range, loss, label='Training loss')
        plt.plot(epoch_range, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
        return plt

    def draw_model_results(self):
        loss, accuracy = self.model.evaluate(self.data.x_image_test, self.data.y_vect_test)
        print("Accuracy for test data : ", accuracy)
        print("Loss for test data : ", loss)
        return loss, accuracy

    def make_predictions(self):
        self.predictions = self.model.predict(self.data.x_image_test, batch_size=64, verbose=1)
        self.predictions = np.argmax(self.predictions, axis=1)

    def draw_confusion_matrix(self):
        self.confusion_matrix = confusion_matrix(self.data.y_test, self.predictions)
        self.confusion_matrix = self.confusion_matrix / self.confusion_matrix.astype(np.float).sum(axis=1)  # normalize
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(self.confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(self.confusion_matrix.shape[0]):
            for j in range(self.confusion_matrix.shape[1]):
                ax.text(x=j, y=i, s=self.confusion_matrix[i, j], va='center', ha='center', size='large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()
        return self.confusion_matrix

    def metric_precision(self):
        acc = {}
        for i in range(len(self.data.label_names)):
            acc[self.data.label_names[i]] = self.confusion_matrix[i][i] / sum(self.confusion_matrix[:, i])
        return pd.DataFrame({'class': acc.keys(), 'precision': acc.values()})

    def metric_recall(self):
        recall = {}
        for i in range(len(self.data.label_names)):
            recall[i] = self.confusion_matrix[i][i] / sum(self.confusion_matrix[i, :])
        return pd.DataFrame({'class': recall.keys(), 'recall': recall.values()})

    def metric_f1(self):
        f1 = {}
        for i in range(len(self.data.label_names)):
            f1[i] = 2 * (self.confusion_matrix[i, i] / sum(self.confusion_matrix[:, i])) \
                    * (self.confusion_matrix[i, i] / sum(self.confusion_matrix[i, :])) / (
                            (self.confusion_matrix[i, i] / sum(self.confusion_matrix[:, i]))
                            + (self.confusion_matrix[i, i] / sum(self.confusion_matrix[i, :])))
        return pd.DataFrame({'class': f1.keys(), 'F1-score': f1.values()})

    def metrics(self):
        accurancy = self.metric_precision()
        recall = self.metric_recall()
        f1 = self.metric_f1()
        return pd.concat([accurancy, recall['recall'], f1['F1-score']], axis=1, join='inner')
