from token_access import *
from data_access import *
from data_process import *


if __name__ == "__main__":
    # copy ur client_id and secret by creating your app
    client_id = '29c900968c9544c6b8ce969b4422d31e'
    client_secret = '28dd01f72a1c4d71b828edb0bdcf79aa'

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
    read_csv_get_strings(filename, token, 18000, 1000)
