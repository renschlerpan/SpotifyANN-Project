import base64
import requests

def get_access_token(client_id, client_secret):
    auth_str = f"{client_id}:{client_secret}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()

    auth_options = {
        'url': 'https://accounts.spotify.com/api/token',
        'headers': {
            'Authorization': 'Basic ' + auth_b64
        },
        'data': {
            'grant_type': 'client_credentials'
        }
    }

    response = requests.post(**auth_options)
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        return None

