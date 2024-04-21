from token_access import *
from data_access import *
from data_process import *


if __name__ == "__main__":
    # copy ur client_id and secret by creating your app
    client_id = 'e8ac083093fa4ceb9d1ba0be51145dcb'
    client_secret = '18c9926a3f354362916c51123790f6a6'

    token = get_access_token(client_id, client_secret)

    if token:
        access_token = token
        # print(token)
    else:
        print("Failed to obtain access token.")
        exit(0)
    #change your file path to the spotify_sim.csv file
    filename = "/Users/allenchien/Downloads/spotify_sim.csv"
    #modify which row to start and which row to end
    read_csv_get_strings(filename, token, 0, 600)
