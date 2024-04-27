from token_access import *
from data_access import *
import csv

def read_csv_get_strings(filename, access_token = None, startrow=0, endrow=0):
    data = []
    df = pd.read_csv(filename, header=None, skiprows=startrow, nrows=endrow,)  # Assuming there is no header and read only first 1000 rows
    count = 0
    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # Extract the first and second columns as strings and append to the data list
        artist_name = str(row[0])
        track_name = str(row[1])
        # print(artist_name, track_name)
        result_js = search_spotify(access_token, track_name)
        if result_js is None:
            continue
        track_id = find_track_id(result_js, artist_name)
        if track_id is None:
            continue
        audio_feature = get_audio_features(track_id, access_token)
        genre = get_genre(track_id, access_token)

        if audio_feature is not None:
            count += 1
            audio_feature.insert(0, 'Track_name', track_name)
            audio_feature.insert(0, 'Artist Name', artist_name)
            # print(audio_feature)
            write_dataframe_to_csv(audio_feature, "audio_features.csv")

    print(count)

    # return data
def write_dataframe_to_csv(df, filename):
    df.to_csv(filename, index=False, header=False, mode='a')


def merge_datasets():
    pass
