import requests
import pandas as pd
import time

def search_spotify(access_token, track_name):
    url = "https://api.spotify.com/v1/search"
    params = {
        "q": track_name,
        "type": "track"
    }
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None


def find_track_id(json_content, artist_name):
    # Extracting the items from the JSON content
    items = json_content['tracks']['items']

    # Loop through the items to find the track by the specified artist
    for item in items:
        # Check if the artist name matches the specified artist
        artists = item['album']['artists']
        for artist in artists:
            if artist['name'] == artist_name:
                # Return the track ID if the artist matches
                return item['id']

    # If no matching track is found, return None
    return None

def get_genre(Artistname, access_token):
    pass

def get_audio_features(track_id, access_token, max_retries = 5):
    # Spotify API endpoint for retrieving audio features
    audio_url = f"https://api.spotify.com/v1/audio-features/{track_id}"

    track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
    # Header with authorization token
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    for retry_count in range(max_retries):
        try:
            # Sending GET request to Spotify API
            audio_response = requests.get(audio_url, headers=headers)
            audio_response.raise_for_status()  # Raise an exception for any HTTP error

            genre_response = requests.get(track_url, headers=headers)
            genre_response.raise_for_status()

            # Parsing JSON response
            audio_data = audio_response.json()
            genre_data = genre_response.json()

            # Check if the response is not empty
            if audio_data:
                # Extracting relevant audio features
                audio_features = {
                    "danceability": audio_data["danceability"],
                    "energy": audio_data["energy"],
                    "key": audio_data["key"],
                    "loudness": audio_data["loudness"],
                    "mode": audio_data["mode"],
                    "speechiness": audio_data["speechiness"],
                    "acousticness": audio_data["acousticness"],
                    "instrumentalness": audio_data["instrumentalness"],
                    "liveness": audio_data["liveness"],
                    "valence": audio_data["valence"],
                    "tempo": audio_data["tempo"],
                    "duration_ms": audio_data["duration_ms"],
                    "time_signature": audio_data["time_signature"]
                }

                # Converting dictionary to pandas DataFrame
                df = pd.DataFrame(audio_features, index=[0])
                print(str(genre_data))
                return df

        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")

        except KeyError as err:
            print(
                f"Key error occurred: {err}. Please verify that the track ID is correct and the API response is as expected.")

        except Exception as err:
            print(f"An error occurred: {err}")

        # Return None if max retries reached without success
    return None