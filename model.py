import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense, BatchNormalization

class SongDataProcessor:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.label_encoder = None
        self.label_encoder_song = None
        self.X_scaler = None
        self.Y_scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def create_model(self):
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(self.X_train.shape[1],)),
            BatchNormalization(),  # Add batch normalization layer
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dense(10, activation='softmax')  # Output layer with softmax activation function
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=100, batch_size=32):
        self.create_model()
        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       validation_data=(self.X_test, self.y_test))

    def predict(self, X):
        # Scale the input data using the same scaler used during preprocessing
        X_scaled = self.scaler.transform(X)

        # Perform predictions using the trained model
        predictions = self.model.predict(X_scaled)

        # Decode the predictions if needed
        decoded_predictions = self.label_encoder_song.inverse_transform(predictions)

        return decoded_predictions

    def load_data(self):
        # Load the CSV file into a DataFrame
        self.data = pd.read_csv(self.csv_file)

    def preprocess_data(self):
        # Encode the 'playlist_genre' column using LabelEncoder
        self.label_encoder = LabelEncoder()
        self.data['playlist_genre'] = self.label_encoder.fit_transform(self.data['playlist_genre'])

        self.label_encoder_song = LabelEncoder()
        self.data['track_name'] = self.label_encoder_song.fit_transform(self.data['track_name'])

        # Separate features (X) and target (y)
        X = self.data.drop(columns=['track_name'])
        # print(X)
        # y = self.get_random_song_names()
        y = []
        # Iterate over each row in the DataFrame
        for index, row in X.iterrows():
            # Generate output based on the input features
            output_song_names = self.get_random_song_names(row['playlist_genre'])
            # Append the output to the target variable y
            row_dict = {'song' + str(i + 1): song_name for i, song_name in enumerate(output_song_names)}

            y.append(row_dict)

        y_df = pd.DataFrame(y)
        # Standardize the features
        self.X_scaler = StandardScaler()
        X_scaled = self.X_scaler.fit_transform(X)
        self.Y_scaler = StandardScaler()
        y_scaled = self.Y_scaler.fit_transform(y_df)

        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        print("Train input shape: ",self.X_train.shape)
        print("Train label shape: ", self.y_train.shape)
        print("Test input shape: ",self.X_test.shape)
        print("Test label shape: ", self.y_test.shape)
        print(y_scaled)
    def get_random_song_names(self, genre):
        # Filter the dataset to include only songs from the specified genre
        # genre_index = self.label_encoder.transform([genre])[0]
        genre_songs = self.data[self.data['playlist_genre'] == genre]

        # Select 10 random song names from the filtered dataset
        random_song_names = random.sample(list(genre_songs['track_name']), min(10, len(genre_songs)))
        return random_song_names

    def decode_songname(self, label_encoder_song, encoded_song_names):
        encoded_song_names = self.Y_scaler.inverse_transform(encoded_song_names)
        decoded_song_names = label_encoder_song.inverse_transform(encoded_song_names)
        return decoded_song_names
