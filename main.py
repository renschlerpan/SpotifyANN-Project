from token_access import *
from data_access import *

def remove_first_last_column(csv_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Remove the second and third column
    df = df.drop(columns=[df.columns[1], df.columns[2]])

    # Save the modified DataFrame back to a new CSV file
    df.to_csv(output_file, index=False)



if __name__ == "__main__":
    # client_id = '5da8e50fe69b4bc0a53c5fa8874bf577'
    # client_secret = '83aedf5f2e0f43ac910cd7f41cd15d2c'
    #
    # token = get_access_token(client_id, client_secret)
    #
    # if token:
    #     access_token = token
    #     print("Access token:", token)
    # else:
    #     print("Failed to obtain access token.")
    #     exit(0)
    # result_js = search_spotify(access_token)
    # track_id = find_track_id(result_js, "Paul McCartney")
    # audio_feature = get_audio_features(track_id, access_token)
    # print(audio_feature)
    # Example usage:
    input_csv_file = "/Users/allenchien/Downloads/archive/spotify_dataset.csv"  # Replace with the path to your input CSV file
    output_csv_file = "output.csv"  # Replace with the desired path for the output CSV file

    remove_first_last_column(input_csv_file, output_csv_file)

