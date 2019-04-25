import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Input
from keras.models import Model, Sequential
from keras.utils import to_categorical
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation, Conv1D, MaxPool1D, UpSampling1D
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers


def plot_history(history, title):
    """
    Plots the  training and validation accuracy and loss in two graphs
    :param history: History of classifier during training
    :param title:  Title displayed at top graph
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(12, 5))
    plt.plot(loss, 'b', label='Training loss')
    plt.plot(val_loss, 'r', label='Validation loss')
    plt.title('{} Training and validation loss'.format(title))
    plt.legend()
    plt.show()


def cluster_autoencoder_deep(train_data, train_label, num_classes, title, num_epochs):
    """
    Clusters assignment  using the deep autoencoder
    :param train_data: Data used to train autoencoder
    :param train_label:  Labels used for evaluating cluster assignments
    :param num_classes: Number of classes inside data and number clusters
    :param title: Title used in plotting graph
    :param num_epochs: Number of epochs in autoencoder
    :return: Evalaution results on three metrics
    """

    batch_size = 128
    num_neurons_input = num_neurons_output = train_data.shape[1]  # number features
    # splits data into training and validation data
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1,
                                                                    random_state=RANDOM_STATE)

    # encoder
    input_shape = Input(shape=(num_neurons_input,))
    encoded = Dense(128, activation='relu')(input_shape)
    encoded = Dense(65, activation='relu')(encoded)
    encoded = Dense(35, activation='relu')(encoded)
    encoded = Dense(num_classes, activation='linear', activity_regularizer=regularizers.l2(10e-5))(encoded)

    #decoder
    decoded = Dense(35, activation='relu')(encoded)
    decoded = Dense(65, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(units=num_neurons_output, activation="softmax")(decoded)
    autoencoder = Model(input_shape, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer="adam", loss="mse")
    history = autoencoder.fit(train_data, train_data, batch_size=batch_size, epochs=num_epochs,
                              validation_data=(val_data, val_data), verbose=0)
    plot_history(history, title) # plots teaing and validation loss

    encoder = Model(input_shape, encoded)
    encoded_data = encoder.predict(train_data)
    softmax_predictions = cluster_assignment(encoded_data) # predictions from softmax layer
    # cluster assignments
    cluster_prediction = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE).fit_predict(softmax_predictions)

    # evaluation
    homo = homogeneity_score(train_label, cluster_prediction)
    comp = completeness_score(train_label, cluster_prediction)
    v = v_measure_score(train_label, cluster_prediction)

    return "homo: {} comp: {} v-Measure: {} ".format(homo, comp, v)


def cluster_autoencoder_simple(train_data, train_label, num_classes, title, num_epochs):
    """
    Clusters assignment using the simple autoencoder
    :param train_data: Data used to train autoencoder
    :param train_label:  Labels used for evaluating cluster assignments
    :param num_classes: Number of classes inside data and number clusters
    :param title: Title used in plotting graph
    :param num_epochs: Number of epochs in autoencoder
    :return: Evalaution results on three metrics
    """
    batch_size = 128
    num_neurons_input = num_neurons_output = train_data.shape[1]  # number features
    # splits data into training and validation data
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1,
                                                                    random_state=RANDOM_STATE)

    input_shape = Input(shape=(num_neurons_input,))
    encoded = Dense(50, activation='relu')(input_shape)
    decoded = Dense(num_neurons_output, activation='sigmoid')(encoded)
    autoencoder = Model(input_shape, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer="adam", loss="mse")
    history = autoencoder.fit(train_data, train_data, batch_size=batch_size, epochs=num_epochs,
                              validation_data=(val_data, val_data), verbose=0)
    plot_history(history, title) # plots training and validation loss

    # encodes and clusters data
    encoder = Model(input_shape, encoded)
    encoded_data = encoder.predict(train_data)
    cluster_prediction = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE).fit_predict(encoded_data)

    # evaluation
    homo = homogeneity_score(train_label, cluster_prediction)
    comp = completeness_score(train_label, cluster_prediction)
    v = v_measure_score(train_label, cluster_prediction)

    return "homo: {} comp: {} v-Measure: {} ".format(homo, comp, v)


def cluster_assignment(encoded_data):
    """
    Converts softmax layer data to cluster assignments
    :param encoded_data: Softmax layer outputs
    :return: Cluster assignments from highest value neurons
    """
    assignments = []
    for data in encoded_data:
        assignments.append([data.argmax()])
    return assignments


def cluster_none(train_data, train_labels, num_classes):
    """
    Clusters data using no autoencoder
    :param train_data: Data used to train autoencoder
    :param train_labels:  Labels used for evaluating cluster assignments
    :param num_classes: Number of classes inside data and number clusters
    :return: evalaution results
    """
    clusters = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE).fit_predict(train_data)
    homo = homogeneity_score(train_labels, clusters)
    comp = completeness_score(train_labels, clusters)
    v = v_measure_score(train_labels, clusters)
    return "homo: {} comp: {} v-Measure: {} ".format(homo, comp, v)


def cluster_PCA(train_data, train_label, num_classes, components):
    """
    clusters the data using PCA compoenets
    :param train_data: Data used to train autoencoder
    :param train_label:  Labels used for evaluating cluster assignments
    :param num_classes: Number of classes inside data and number clusters
    :param components: number of PCA compoenets
    :return: evaluation results
    """
    pca = PCA(components)
    pca.fit(train_data)

    # plots expected variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Dataset Explained Variance')
    plt.show()

    train_data_pca = pca.transform(train_data)
    clusters = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE).fit_predict(train_data_pca)

    # evaluation
    homo = homogeneity_score(train_label, clusters)
    comp = completeness_score(train_label, clusters)
    v = v_measure_score(train_label, clusters)

    return "homo: {} comp: {} v-Measure: {} ".format(homo, comp, v)


# filepaths of data
CNAE_9_PATH = "CNAE_9.data"
HAR_DATA_PATH = "HAR_data.txt"
HAR_LABEL_PATH = "HAR_label.txt"
DIGIT_PATH = "digit_data.csv"
RANDOM_STATE = 2472

NUM_CNAE = 9
NUM_HAR = 6
NUM_DIGIT = 10

# reads cnae data
cnae = pd.read_csv(CNAE_9_PATH, header=None)
cnae_data = cnae.drop(cnae.columns[0], axis=1)
cnae_labels = cnae[cnae.columns[0]]

# reads har data
har_data = pd.read_csv(HAR_DATA_PATH, delim_whitespace=True)
har_labels = pd.read_csv(HAR_LABEL_PATH, delim_whitespace=True)
har_labels = har_labels.squeeze()

# reads digit data
digit = pd.read_csv(DIGIT_PATH)
digit_data = digit.drop(digit.columns[0], axis=1)
digit_labels = digit[digit.columns[0]]


# Cluster dataset using original none-scaled features
cnae_none = cluster_none(cnae_data, cnae_labels, NUM_CNAE)
har_none = cluster_none(har_data, har_labels, NUM_HAR)
digit_none = cluster_none(digit_data, digit_labels, NUM_DIGIT)

# cluster using PCA Features with normal data
digit_PCA = cluster_PCA(digit_data, digit_labels, NUM_DIGIT, 130)
cnae_PCA = cluster_PCA(cnae_data, cnae_labels, NUM_CNAE, 250)
har_PCA = cluster_PCA(har_data, har_labels, NUM_HAR, 60)

# normalises data using min max
cnae_data_mm = MinMaxScaler().fit_transform(cnae_data)
digit_data_mm = MinMaxScaler().fit_transform(digit_data)
har_data_mm = MinMaxScaler().fit_transform(har_data)

# cluster using deep autoenocder with normalised data
digit_results_mm = cluster_autoencoder_deep(digit_data_mm, digit_labels, NUM_DIGIT, "MNIST digit", 200)
cnae_results_mm = cluster_autoencoder_deep(cnae_data_mm, cnae_labels, NUM_CNAE, "CNAE-9", 200)
har_results_mm = cluster_autoencoder_deep(har_data_mm, har_labels, NUM_HAR, "Human activity", 200)

# cluster using simple autoencoder with normalised data
digit_simple_mm = cluster_autoencoder_simple(digit_data_mm, digit_labels, NUM_DIGIT, "MNIST digit", 200)
cnae_simple_mm = cluster_autoencoder_simple(cnae_data_mm, cnae_labels, NUM_CNAE, "CNAE-9", 200)
har_simple_mm = cluster_autoencoder_simple(har_data_mm, har_labels, NUM_HAR, "Human activity", 200)

# cluster using no autoenocder with normalised data
cnae_none_mm = cluster_none(cnae_data_mm, cnae_labels, NUM_CNAE)
har_none_mm = cluster_none(har_data_mm, har_labels, NUM_HAR)
digit_none_mm = cluster_none(digit_data_mm, digit_labels, NUM_DIGIT)

# cluster pca with normalised data
digit_PCA_mm = cluster_PCA(digit_data_mm, digit_labels, NUM_DIGIT, 130)
cnae_PCA_mm = cluster_PCA(cnae_data_mm, cnae_labels, NUM_CNAE, 250)
har_PCA_mm = cluster_PCA(har_data_mm, har_labels, NUM_HAR, 60)

# prints clustering with data
print("CNAE none", cnae_none)
print("HAR none", har_none)
print("Digit none", digit_none, '\n')

# prints PCA with normal data
print("CNAE with PCA ", cnae_PCA)  # 120- 30
print("HAR with PCA", har_PCA)  # 230-50
print("Digit with PCA", digit_PCA)  # 60 pca

# prints clustering with normalised data
print("CNAE none mm", cnae_none_mm)
print("HAR none mm", har_none_mm)
print("Digit none mm", digit_none_mm, '\n')

# prints clustering with simple autoencoder
print("CNAE simple mm", cnae_simple_mm)
print("HAR simple mm", har_simple_mm)
print("Digit simple mm", digit_simple_mm, '\n')

# prints clustering with deep autoencoder
print("CNAE Deep mm", cnae_results_mm)
print("HAR Deep mm", har_results_mm)
print("Digit Deep mm", digit_results_mm, '\n')

# prints clustering with normalised PCA
print("CNAE with PCA MM", cnae_PCA_mm)  # 120- 30
print("HAR with PCA MM", har_PCA_mm)  # 230-50
print("Digit with PCA MM", digit_PCA_mm)  # 60 pca
