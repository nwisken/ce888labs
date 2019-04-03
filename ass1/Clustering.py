import pandas as pd
from keras import Input
from keras.layers import Dense
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.decomposition import PCA
import matplotlib as plt
import numpy as np

def cluster_autoencoder(train_data, train_label, num_classes):
    batch_size = 128
    num_epochs = 50
    num_neurons_input = num_neurons_output = train_data.shape[1]
    # num_neurons_hl1 = 35
    num_neurons_hl1 = 50

    input_shape = Input(shape=(num_neurons_input,))
    # Dense = NN network layer
    encoded = Dense(units=num_neurons_hl1, activation="relu")(input_shape)
    decoded = Dense(units=num_neurons_output, activation="sigmoid")(encoded)
    autoencoder = Model(input_shape, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer="adadelta", loss="mse")
    autoencoder.fit(train_data, train_data, batch_size=batch_size, epochs=num_epochs, shuffle=True)

    encoder = Model(input_shape, encoded)
    encoded_data = encoder.predict(train_data)

    cluster_preicdion = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE).fit(encoded_data)

    homo = homogeneity_score(train_label, cluster_preicdion.labels_)
    comp = completeness_score(train_label, cluster_preicdion.labels_)
    v = v_measure_score(train_label, cluster_preicdion.labels_)

    return "homo: {} comp: {} v-Measure: {} ".format(homo, comp, v)


def cluster_PCA(train_data, train_label, num_classes):
    pca = PCA(.95)
    pca.fit(train_data)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Pulsar Dataset Explained Variance')
    plt.show()

    train_data_pca = pca.transform(train_data)
    clusters = KMeans(n_clusters=num_classes, random_state=RANDOM_STATE).fit(train_data_pca)

    homo = homogeneity_score(train_label, clusters.labels_)
    comp = completeness_score(train_label, clusters.labels_)
    v = v_measure_score(train_label, clusters.labels_)

    return "homo: {} comp: {} v-Measure: {} ".format(homo, comp, v)


CNAE_9_PATH = "CNAE_9.data"
HAR_DATA_PATH = "HAR_data.txt"
HAR_LABEL_PATH = "HAR_label.txt"
DIGIT_PATH = "digit_data.csv"
RANDOM_STATE = 2472

NUM_EPOCHS = 100
BATCH_SIZE = 256

cnae = pd.read_csv(CNAE_9_PATH, header=None)
cnae_data = cnae.drop(cnae.columns[0], axis=1)
cnae_label = cnae[cnae.columns[0]]

har_data = pd.read_csv(HAR_DATA_PATH, delim_whitespace=True)
har_labels = pd.read_csv(HAR_LABEL_PATH, delim_whitespace=True)
har_labels = har_labels.squeeze()

digit = pd.read_csv(DIGIT_PATH)
digit_data = digit.drop(digit.columns[0], axis=1)
digit_label = digit[digit.columns[0]]

cnae_clusters = KMeans(n_clusters=9, random_state=RANDOM_STATE).fit(cnae_data)
cnae_homo = homogeneity_score(cnae_label, cnae_clusters.labels_)
cnae_comp = completeness_score(cnae_label, cnae_clusters.labels_)
cnae_v = v_measure_score(cnae_label, cnae_clusters.labels_)

#print("CNAE homo: {} comp: {} v-Measure: {} ".format(cnae_homo, cnae_comp, cnae_v))

har_clusters = KMeans(n_clusters=6, random_state=RANDOM_STATE).fit(har_data)
har_homo = homogeneity_score(har_labels, har_clusters.labels_)
har_comp = completeness_score(har_labels, har_clusters.labels_)
har_v = v_measure_score(har_labels, har_clusters.labels_)

#print("HAR homo: {} comp: {} v-Measure: {} ".format(har_homo, har_comp, har_v))

digit_clusters = KMeans(n_clusters=10, random_state=RANDOM_STATE).fit(digit_data)
digit_homo = homogeneity_score(digit_label, digit_clusters.labels_)
digit_comp = completeness_score(digit_label, digit_clusters.labels_)
digit_v = v_measure_score(digit_label, digit_clusters.labels_)

#print("Digit homo: {} comp: {} v-Measure: {} ".format(digit_homo, digit_comp, digit_v))

# digit_results = cluster_autoencoder(digit_data, digit_label, 10)
# cnae_results = cluster_autoencoder(cnae_data, cnae_label, 9)
# har_results = cluster_autoencoder(har_data, har_labels, 6)

digit_PCA = cluster_PCA(digit_data, digit_label, 10)
cnae_PCA = cluster_PCA(cnae_data, cnae_label, 9)
har_PCA = cluster_PCA(har_data, har_labels, 6)

print("CNAE homo: {} comp: {} v-Measure: {} ".format(cnae_homo, cnae_comp, cnae_v))
print("HAR homo: {} comp: {} v-Measure: {} ".format(har_homo, har_comp, har_v))
print("Digit homo: {} comp: {} v-Measure: {} ".format(digit_homo, digit_comp, digit_v))
# print("CNAE with autoencoder", cnae_results)
# print("HAR with autoencoder", har_results)
# print("Digit with autoencoder", digit_results)
print("CNAE with PCA", cnae_PCA)
print("HAR with PCA", har_PCA)
print("Digit with PCA", digit_PCA)
