"""Download data from Google Drive either by copying it from Google Drive to Google Colab or by using gdown."""
import os
import shutil
import subprocess
from typing import Optional

from src import IS_COLAB


def _download_data(destination_path: str, file_id: str, force: bool = False) -> None:
    """Download data from Google Drive using gdown.
    :param destination_path: path to download data to
    :type destination_path: str
    :param file_id: id of the file to download
    :type file_id: str
    :param force: whether to force downloading data
    :type force: bool
    :return: None
    :rtype: None
    """
    if not os.path.exists(destination_path) or force:
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


def _copy_from_drive(destination_path: str, source_path: str, force: bool = False) -> None:
    """Copies the data from google drive. Function used in Google Colab.
    :param destination_path: path to download data to
    :type destination_path: str
    :param source_path: path to download data from
    :type source_path: str
    :param force: whether to force copying data
    :type force: bool
    :return: None
    """
    if not os.path.exists(destination_path) or force:
        from google.colab import drive

        drive.mount("/content/drive")
        if not os.path.exists(source_path):
            raise ValueError(f"File {source_path} does not exist!")
        shutil.copy(source_path, "data.zip")
        # unzip data
        subprocess.run(["unzip", "-q", "data.zip", "-d", destination_path])
        # remove zip file
        os.remove("data.zip")


def download_dataset(
    destination_path: str = "../../",
    source_path: Optional[str] = "drive/MyDrive/cityscapes_dataset.zip",
    file_id: Optional[str] = "1w2khj04-wdtAj-1dNLWWKZhVo26BjMG2",
    force: bool = False,
) -> None:
    """Download data using gdown, when using local machine, copy from Google Drive when using Google Colab.
    :param destination_path: path to download data to
    :type destination_path: str
    :param source_path: path to download data from
    :type source_path: str
    :param file_id: id of the file to download
    :type file_id: str
    :param force: whether to force downloading data
    :type force: bool
    :return: None
    :rtype: None
    """
    if IS_COLAB:
        assert source_path is not None, "source_path cannot be None when copying from Google Drive"
        _copy_from_drive(destination_path=destination_path, source_path=source_path, force=force)
    else:
        assert file_id is not None, "source_path cannot be None when downloading from GDrive"
        _download_data(destination_path=destination_path, file_id=file_id, force=force)


if __name__ == "__main__":
    download_dataset()
