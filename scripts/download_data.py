import os
import shutil
import subprocess

from src import IS_COLAB


def download_data(destination_path: str, file_id:str) -> None:
    """Download data from Google Drive using gdown.
    :param destination_path: path to download data to
    :type destination_path: str
    :param file_id: id of the file to download
    :type file_id: str
    :return: None
    :rtype: None
    """
    import gdown

    # create data directory
    os.makedirs(destination_path, exist_ok=True)
    # download data
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join(destination_path, "data.zip")
    gdown.download(url, output, quiet=False)
    # unzip data
    subprocess.run(["unzip", "-q", output, "-d", destination_path])
    # remove zip file
    os.remove(output)


def copy_from_drive(destination_path: str, source_path: str) -> None:
    """Copies the data from google drive. Function used in Google Colab.
    :param destination_path: path to download data to
    :type destination_path: str
    :param source_path: path to download data from
    :type source_path: str
    """
    from google.colab import drive

    if not os.path.exists(source_path):
        raise ValueError(f"File {source_path} does not exist!")

    drive.mount("/content/drive")
    shutil.copy(source_path, "data.zip")
    # unzip data
    subprocess.run(["unzip", "-q", "data.zip", "-d", destination_path])
    # remove zip file
    os.remove("data.zip")


def download_dataset() -> None:
    """Download data using gdown, when using local machine, copy from Google Drive when using Google Colab.
    :return: None
    :rtype: None
    """
    if IS_COLAB:
        copy_from_drive(destination_path="../", source_path="drive/MyDrive/cityscapes_dataset.zip")
    else:
        download_data(destination_path="../", file_id="1w2khj04-wdtAj-1dNLWWKZhVo26BjMG2")


if __name__ == "__main__":
    download_dataset()