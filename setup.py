import os

from setuptools import find_packages, setup

from scripts.get_project_version import get_project_version


def read(file_name: str) -> str:
    """Read the file and return its content as a string.
    :param file_name: name of the file to read
    :rtype file_name: str
    :return: content of the file
    :rtype: str
    """
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


def read_requirements(file_name: str) -> list:
    """Read the file and return its content as a list.
    :param file_name: name of the file to read
    :rtype file_name: str
    :return: content of the file
    :rtype: list
    """
    with open(os.path.join(os.path.dirname(__file__), file_name), "r") as f:
        required = f.read().splitlines()
    return required


setup(
    name="depth_estimation",
    version=get_project_version(),
    author="Maciej Filanowicz & Mikołaj Kruś",
    description="Tools used in the project dedicated to depth estimation",
    packages=find_packages(include=["src", "src.*"]),
    long_description=read("README.md"),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "isort~=5.10.1",
            "black~=22.6.0",
            "flake8~=5.0.4",
            "pylint~=2.14.3",
            "mypy~=0.971",
            "jupyter==1.0.0"
            "pytest==7.3.1"
            "sphinx-rtd-theme==1.2.0"

        ]
    },
    include_package_data=True,
    package_data={"src": ["*.yaml"]},
)
