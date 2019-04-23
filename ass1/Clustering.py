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


def plot_history(history, title):
    """
    Plots the  training and validation accuracy and loss in two graphs
    :param history: History of classifier during training
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # x = range(NUM_EPOCHS)

    plt.figure(figsize=(12, 5))
    plt.plot(loss, 'b', label='Training loss')
    plt.plot(val_loss, 'r', label='Validation loss')
    plt.title('{} Training and validation loss'.format(title))
    plt.legend()
    plt.show()


def cluster_autoencoder(train_data, train_label, num_classes, title, num_epochs):
    batch_size = 128
    num_neurons_input = num_neurons_output = train_data.shape[1]  # number features
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1,
                                                                    random_state=RANDOM_STATE)

    input_shape = Input(shape=(num_neurons_input,))
    # Dense = NN network layer
    encoded = Dense(128, activation='relu')(input_shape)
    encoded = Dense(65, activation='relu')(encoded)
    encoded = Dense(35, activation='relu')(encoded)

    decoded = Dense(65, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(units=num_neurons_output, activation="sigmoid")(decoded)
    autoencoder = Model(input_shape, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    history = autoencoder.fit(train_data, train_data, batch_size=batch_size, epochs=num_epochs,
                              validation_data=(val_data, val_data), verbose=0)
    plot_history(history, title)

    classifier = Dense(num_classes, activation="softmax")(encoded)
    encoder = Model(input_shape, encoded)
    encoded_data = encoder.predict(train_data)

    cluster_preicdion = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE).fit_predict(encoded_data)

    homo = homogeneity_score(train_label, cluster_preicdion)
    comp = completeness_score(train_label, cluster_preicdion)
    v = v_measure_score(train_label, cluster_preicdion)

    return "homo: {} comp: {} v-Measure: {} ".format(homo, comp, v)


def cluster_autoencoder_simple(train_data, train_label, num_classes, title, num_epochs):
    batch_size = 128
    num_neurons_input = num_neurons_output = train_data.shape[1]  # number features
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1,
                                                                    random_state=RANDOM_STATE)

    input_shape = Input(shape=(num_neurons_input,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(50, activation='relu')(input_shape)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(num_neurons_output, activation='sigmoid')(encoded)

    autoencoder = Model(input_shape, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    history = autoencoder.fit(train_data, train_data, batch_size=batch_size, epochs=num_epochs,
                              validation_data=(val_data, val_data), verbose=0, shuffle=True)
    plot_history(history, title)

    encoder = Model(input_shape, encoded)
    encoded_data = encoder.predict(train_data)

    cluster_preicdion = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE).fit_predict(encoded_data)

    homo = homogeneity_score(train_label, cluster_preicdion)
    comp = completeness_score(train_label, cluster_preicdion)
    v = v_measure_score(train_label, cluster_preicdion)

    return "homo: {} comp: {} v-Measure: {} ".format(homo, comp, v)


def cluster_none(train_data, train_labels, num_classes):
    clusters = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE).fit_predict(train_data)
    homo = homogeneity_score(train_labels, clusters)
    comp = completeness_score(train_labels, clusters)
    v = v_measure_score(train_labels, clusters)
    return "homo: {} comp: {} v-Measure: {} ".format(homo, comp, v)


def cluster_PCA(train_data, train_label, num_classes, components):
    pca = PCA(components)
    pca.fit(train_data)

    """
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Dataset Explained Variance')
    plt.show()
    """

    train_data_pca = pca.transform(train_data)
    clusters = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE).fit_predict(train_data_pca)

    homo = homogeneity_score(train_label, clusters)
    comp = completeness_score(train_label, clusters)
    v = v_measure_score(train_label, clusters)

    return "homo: {} comp: {} v-Measure: {} ".format(homo, comp, v)


CNAE_9_PATH = "CNAE_9.data"
HAR_DATA_PATH = "HAR_data.txt"
HAR_LABEL_PATH = "HAR_label.txt"
DIGIT_PATH = "digit_data.csv"
RANDOM_STATE = 2472

NUM_EPOCHS = 100
BATCH_SIZE = 256

NUM_CNAE = 9
NUM_HAR = 6
NUM_DIGIT = 10

cnae = pd.read_csv(CNAE_9_PATH, header=None)
cnae_data = cnae.drop(cnae.columns[0], axis=1)
cnae_labels = cnae[cnae.columns[0]]

har_data = pd.read_csv(HAR_DATA_PATH, delim_whitespace=True)
har_labels = pd.read_csv(HAR_LABEL_PATH, delim_whitespace=True)
har_labels = har_labels.squeeze()

digit = pd.read_csv(DIGIT_PATH)
digit_data = digit.drop(digit.columns[0], axis=1)
digit_labels = digit[digit.columns[0]]

"""
# Cluster dataset using original none-scaled features
cnae_none = cluster_none(cnae_data, cnae_labels, NUM_CNAE)
har_none = cluster_none(har_data, har_labels, NUM_HAR)
digit_none = cluster_none(digit_data, digit_labels, NUM_DIGIT)

# cluster using simple autoencoder
digit_simple= cluster_autoencoder_simple(digit_data, digit_labels, 10,"MNIST digit",50)
cnae_simple = cluster_autoencoder_simple(cnae_data, cnae_labels, 9,"CNAE-9", 50)
har_simple = cluster_autoencoder_simple(har_data, har_labels, 6,"Human activity",50)

# cluster using deep autoenocder
digit_results = cluster_autoencoder(digit_data, digit_labels, 10,"MNIST digit",30)
cnae_results = cluster_autoencoder(cnae_data, cnae_labels, 9,"CNAE-9",4)
har_results = cluster_autoencoder(har_data, har_labels, 6,"Human activity",5)

# cluster using PCA Features
digit_PCA = cluster_PCA(digit_data, digit_labels, 10, 130)
cnae_PCA = cluster_PCA(cnae_data, cnae_labels, 9, 250)
har_PCA = cluster_PCA(har_data, har_labels, 6, 60)
"""

cnae_data_mm = MinMaxScaler().fit_transform(cnae_data)
digit_data_mm = MinMaxScaler().fit_transform(digit_data)
har_data_mm = MinMaxScaler().fit_transform(har_data)

digit_results_mm = cluster_autoencoder(digit_data, digit_labels, 10,"MNIST digit",30)
cnae_results_mm = cluster_autoencoder(cnae_data, cnae_labels, 9,"CNAE-9",30)
har_results_mm = cluster_autoencoder(har_data, har_labels, 6,"Human activity",30)
"""
cnae_none_mm = cluster_none(cnae_data_mm, cnae_labels, NUM_CNAE)
har_none_mm = cluster_none(har_data_mm, har_labels, NUM_HAR)
digit_none_mm = cluster_none(digit_data_mm, digit_labels, NUM_DIGIT)

digit_simple_mm = cluster_autoencoder_simple(digit_data_mm, digit_labels, 10, "MNIST digit", 20)
cnae_simple_mm = cluster_autoencoder_simple(cnae_data_mm, cnae_labels, 9, "CNAE-9", 20)
har_simple_mm = cluster_autoencoder_simple(har_data_mm, har_labels, 6, "Human activity", 20)

digit_PCA_mm = cluster_PCA(digit_data_mm, digit_labels, 10, 130)
cnae_PCA_mm = cluster_PCA(cnae_data_mm, cnae_labels, 9, 250)
har_PCA_mm = cluster_PCA(har_data_mm, har_labels, 6, 60)
"""

"""
print("CNAE none", cnae_none)
print("HAR none", har_none)
print("Digit none", digit_none, '\n')

print("CNAE simple", cnae_simple)
print("HAR simple", har_simple)
print("Digit simple", digit_simple,'\n')

print("CNAE with PCA ", cnae_PCA)  # 120- 30
print("HAR with PCA", har_PCA)  # 230-50
print("Digit with PCA", digit_PCA)  # 60 pca

print("CNAE with autoencoder", cnae_results)
#print("HAR with autoencoder", har_results)
#print("Digit with autoencoder", digit_results,'\n')
"""

# print("CNAE none mm", cnae_none_mm)
# print("HAR none mm", har_none_mm)
# print("Digit none mm", digit_none_mm, '\n')

#print("CNAE simple mm", cnae_simple_mm)
#print("HAR simple mm", har_simple_mm)
#print("Digit simple mm", digit_simple_mm, '\n')

print("CNAE Deep mm", cnae_results_mm)
print("HAR Deep mm", har_results_mm)
print("Digit Deep mm", digit_results_mm, '\n')

# print("CNAE with PCA MM", cnae_PCA_mm)  # 120- 30
# print("HAR with PCA MM", har_PCA_mm)  # 230-50
# print("Digit with PCA MM", digit_PCA_mm)  # 60 pca
