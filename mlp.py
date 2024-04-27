import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class Autoencoder(tf.keras.Model):
    def __init__(self, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.input_layer = Input(shape=(encoding_dim,))
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Create the full model
        self.model_full = Model(self.input_layer, self.decoder)
        optimizer = Adam(learning_rate=0.004)
        self.model_full.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def build_encoder(self):
        encoder_layer1 = Dense(8, activation='tanh')(self.input_layer)
        encoder_layer2 = Dense(6, activation='tanh')(encoder_layer1)
        bottleneck_layer = Dense(2, activation='tanh', name='bottleneck_layer')(encoder_layer2)
        return Model(self.input_layer, bottleneck_layer)

    def build_decoder(self):
        decoder_layer1 = Dense(8, activation='tanh')(self.encoder.output)
        decoder_layer2 = Dense(6, activation='tanh')(decoder_layer1)
        decoder_output = Dense(self.encoding_dim, activation='softmax')(decoder_layer2)
        return decoder_output

    def train(self, X_train, X_test, epochs=5, batch_size=16):
        X = self.min_max_scale(X_train)
        X_test = self.min_max_scale(X_test)
        history = self.model_full.fit(X, X, epochs=epochs, batch_size=batch_size,
                                      shuffle=True, validation_data=(X_test, X_test))
        print("Training Accuracy:", history.history['accuracy'][-1])
        print("Validation Accuracy:", history.history['val_accuracy'][-1])

    def extract_bottleneck(self, X_inference):
        bottleneck_output = self.model_full.get_layer('bottleneck_layer').output
        model_bottleneck = Model(inputs=self.model_full.input, outputs=bottleneck_output)
        bottleneck_predictions = model_bottleneck.predict(self.min_max_scale(X_inference))
        return bottleneck_predictions

    def cluster_bottleneck_knn(self, bottleneck_data, n_neighbors=5):
        X = bottleneck_data

        kmeans = KMeans(n_clusters=5)
        kmeans.fit(X)
        prediction = self.find_nearest_values(kmeans, X[0], X)
        print(prediction)
        print(self.decoder(prediction[0]).numpy())
        # y_kmeans = kmeans.predict(X)
        # centers = kmeans.cluster_centers_
        # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
        # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
        # plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.title('K-means Clustering')
        # plt.show()

    from sklearn.metrics import pairwise_distances_argmin_min

    def find_nearest_values(self, kmeans_model, data_point, X):
        # Find the centroid to which the data point belongs
        centroid_idx = kmeans_model.predict(data_point.reshape(1, -1))

        # Find all data points assigned to the same centroid
        cluster_points = X[kmeans_model.labels_ == centroid_idx]

        # Calculate distances between the given data point and all points in the cluster
        distances = pairwise_distances_argmin_min(data_point.reshape(1, -1), cluster_points)[1]

        # Sort distances and select the 10 nearest values
        nearest_indices = np.argsort(distances)[:10]
        nearest_values = cluster_points[nearest_indices]

        return nearest_values
    def min_max_scale(self, sequence):
        min_val = np.min(sequence)
        max_val = np.max(sequence)
        scaled_sequence = (sequence - min_val) / (max_val - min_val)
        return scaled_sequence

# Assuming you have your data loaded as 'train_data'
# train_data.shape should be (num_samples, 14)
