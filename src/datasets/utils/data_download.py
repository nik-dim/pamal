import gzip
import shutil
import urllib.request

import requests
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_file_from_https(url, local_filename, message):
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        print(message)
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def _download(url, destination, description="File Download"):
    """This may fail and end up with HTTP 403: Forbidden. This happens
    because the server blocks the download to avoid crawling. Retry if it fails.
    """
    opener = urllib.request.URLopener()
    opener.addheader("User-Agent", "Mozilla/5.0")
    while True:
        try:
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=description) as t:
                opener.retrieve(url, destination, reporthook=t.update_to)
            break
        except:
            print("Download failed. Retrying")


def fix_path(path):
    if path[-1] == "/":
        path = path[:-1]
    return path


def unzip(src, dst):
    with gzip.open(src, "rb") as f_in:
        with open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


import requests


# THE BELOW ARE TAKEN FROM STACKOVERFLOW
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
