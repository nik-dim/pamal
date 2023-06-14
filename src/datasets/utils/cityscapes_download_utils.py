import hashlib
import json
import os
import shutil
from typing import Any

import appdirs
import requests
import tqdm


# username and password used to access https://www.cityscapes-dataset.com/downloads/
CITYSCAPES_USER = "username"
CITYSCAPES_PASSWORD = "password"


def login() -> requests.Session:
    print("Login started for cityscapes with the predefined credentials")

    appname = __name__.split(".")[0]
    appauthor = "cityscapes"
    data_dir = appdirs.user_data_dir(appname, appauthor)
    credentials_file = os.path.join(data_dir, "credentials.json")

    if os.path.isfile(credentials_file):
        with open(credentials_file, "r") as f:
            credentials = json.load(f)
    else:
        username = CITYSCAPES_USER
        password = CITYSCAPES_PASSWORD

        credentials = {"username": username, "password": password}

    session = requests.Session()
    r = session.get("https://www.cityscapes-dataset.com/login", allow_redirects=False)
    r.raise_for_status()

    credentials["submit"] = "Login"
    r = session.post("https://www.cityscapes-dataset.com/login", data=credentials, allow_redirects=False)
    r.raise_for_status()

    # login was successful, if user is redirected
    if r.status_code != 302:
        if os.path.isfile(credentials_file):
            os.remove(credentials_file)
        raise Exception("Bad credentials.")

    return session


def get_available_packages(*, session: requests.Session) -> Any:
    r = session.get("https://www.cityscapes-dataset.com/downloads/?list", allow_redirects=False)
    r.raise_for_status()
    return r.json()


def download_packages(*, session, package_names, destination_path, resume=False) -> None:
    if not os.path.isdir(destination_path):
        raise Exception("Destination path '{}' does not exist.".format(destination_path))

    packages = get_available_packages(session=session)
    name_to_id = {p["name"]: p["packageID"] for p in packages}
    invalid_names = [n for n in package_names if n not in name_to_id]

    if invalid_names:
        raise Exception("These packages do not exist or you don't have access: {}".format(invalid_names))

    for package_name in tqdm.tqdm(package_names):
        local_filename = os.path.join(destination_path, package_name)
        package_id = name_to_id[package_name]

        print("Downloading cityscapes package '{}' to '{}'".format(package_name, local_filename))

        if os.path.exists(local_filename):
            if resume:
                print("Resuming previous download")
            else:
                raise Exception("Destination file '{}' already exists.".format(local_filename))

        # md5sum
        url = "https://www.cityscapes-dataset.com/md5-sum/?packageID={}".format(package_id)
        r = session.get(url, allow_redirects=False)
        r.raise_for_status()
        md5sum = r.text.split()[0]

        # download in chunks, support resume
        url = "https://www.cityscapes-dataset.com/file-handling/?packageID={}".format(package_id)

        with open(local_filename, "ab" if resume else "wb") as f:
            resume_header = {"Range": "bytes={}-".format(f.tell())} if resume else {}

            with session.get(url, allow_redirects=False, stream=True, headers=resume_header) as r:
                r.raise_for_status()
                assert r.status_code in [200, 206]

                shutil.copyfileobj(r.raw, f)

        # verify md5sum
        hash_md5 = hashlib.md5()
        with open(local_filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        if md5sum != hash_md5.hexdigest():
            raise Exception("MD5 sum of downloaded file does not match.")


if __name__ == "__main__":
    session = login()
    path = os.path.expanduser("~/benchmarks/data/cityscapes/")
    print("Downloading Cityscapes dataset to '{}'".format(path))
    download_packages(
        session=session,
        package_names=["gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip"],
        destination_path=path,
        resume=True,
    )
