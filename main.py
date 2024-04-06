import requests

def search_spotify(access_token):
    url = "https://api.spotify.com/v1/search"
    params = {
        "q": "Live And Let Die",
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

if __name__ == "__main__":
    access_token = "BQADjBy4i1FoH5nqU5dqz3d5D8oYG-3FzGKfg13J2UVWH7kelXF_rz30ixPnTY_VGd90jTK-sNolRYAtE02OxYR5vDroeioLbBy-EamY2zTjYd2v5Ks"

    result = search_spotify(access_token)
    if result:
        print(result)
