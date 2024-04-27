from token_access import *
from data_access import *
from data_process import *
from mlp import *
from model import *


if __name__ == "__main__":
    # Create an instance of the class
    data_processor = SongDataProcessor("song data/filtered.csv")
    # Load the data
    data_processor.load_data()
    # Preprocess the data
    data_processor.preprocess_data()
    data_processor.train_model()
    # copy ur client_id and secret by creating your app
    # client_id = '29c900968c9544c6b8ce969b4422d31e'
    # client_secret = '28dd01f72a1c4d71b828edb0bdcf79aa'
    #
    # token = get_access_token(client_id, client_secret)
    #
    # if token:
    #     access_token = token
    #     # print(token)
    # else:
    #     print("Failed to obtain access token.")
    #     exit(0)
    # # change your file path to the spotify_sim.csv file
    # filename = "/Users/allenchien/Downloads/spotify_sim.csv"
    # # modify which row to start and which row to end
    # read_csv_get_strings(filename, token, 22000, 1000)

    # data = pd.read_csv('song data/filtered_spotify.csv', header=None)
    # data = pd.read_csv('audio_features.csv', header=None, on_bad_lines='skip')
    # data = pd.read_csv('song data/filtered.csv', header=None, on_bad_lines='skip')
    #
    #
    # train_data = data.drop(data.columns[0], axis=1)
    # train_data = train_data.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    # # print(train_data)
    #
    # encoding_dim = 12
    # my_autoencoder = Autoencoder(encoding_dim)
    # my_autoencoder.train(train_data, train_data, 3)
    # bottleneck_features = my_autoencoder.extract_bottleneck(train_data)
    # # print(bottleneck_features)
    # my_autoencoder.cluster_bottleneck_knn(bottleneck_features)
    # cluster_labels = my_autoencoder.cluster_bottleneck_knn(bottleneck_features)
    # my_autoencoder.visualize_knn_clusters(bottleneck_features, cluster_labels)
