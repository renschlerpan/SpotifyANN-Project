import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
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
        self.model_full.compile(optimizer=optimizer, loss='mean_squared_error')

    def build_encoder(self):
        encoder_layer1 = Dense(8, activation='tanh')(self.input_layer)
        bottleneck_layer = Dense(4, activation='tanh', name='bottleneck_layer')(encoder_layer1)
        return Model(self.input_layer, bottleneck_layer)

    def build_decoder(self):
        decoder_layer1 = Dense(8, activation='tanh')(self.encoder.output)
        decoder_output = Dense(self.encoding_dim, activation='sigmoid')(decoder_layer1)
        return decoder_output

    def train(self, X_train, X_test, epochs=10, batch_size=100):
        X = self.min_max_scale(X_train)
        y = self.min_max_scale(X_test)
        self.model_full.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                            shuffle=True, validation_data=(X_test, X_test))


    def extract_bottleneck(self, X_inference):
        bottleneck_output = self.model_full.get_layer('bottleneck_layer').output
        model_bottleneck = Model(inputs=self.model_full.input, outputs=bottleneck_output)
        bottleneck_predictions = model_bottleneck.predict(X_inference)
        return bottleneck_predictions

    def cluster_bottleneck_knn(self, bottleneck_data, n_neighbors=5):
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(bottleneck_data)

        # Find nearest neighbors for each data point
        _, indices = knn.kneighbors(bottleneck_data)
        # Assign cluster labels based on nearest neighbors
        cluster_labels = np.mean(indices, axis=1).astype(int)

        return cluster_labels

    def visualize_knn_clusters(self, bottleneck_data, cluster_labels):
        plt.figure(figsize=(8, 6))
        plt.scatter(bottleneck_data[:, 0], bottleneck_data[:, 1], c=cluster_labels, cmap='viridis')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("kNN Clustering")
        plt.colorbar(label="Cluster Label")
        plt.show()

    def min_max_scale(self, sequence):
        min_val = np.min(sequence)
        max_val = np.max(sequence)
        scaled_sequence = (sequence - min_val) / (max_val - min_val)
        return scaled_sequence

# Assuming you have your data loaded as 'train_data'
# train_data.shape should be (num_samples, 14)
