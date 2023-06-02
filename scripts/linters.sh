#!/bin/sh
# Be aware that when running shell script it has default python environment.
# If u wish to run below script with specified python environment please remember to activate it.
# Example is showed below:

main_dir=./src/

echo '<--- Isort ANALYSIS --->'
python3 -m isort --profile black ${main_dir}
echo '<--- BLACK FORMATTING --->'
python3 -m black --line-length 120 --target-version py39 ${main_dir}
echo '<--- Flake8 ANALYSIS --->'
python3 -m flake8 --ignore=E203,E231 --max-line-length 120 --max-complexity 10 ${main_dir}
echo '<--- PYLINT ANALYSIS --->'
PYFILES=$(find ${main_dir} -type f -name "*.py")
python3 -m pylint --rcfile=pylint.ini ${PYFILES}
echo '<--- MYPY ANALYSIS --->'
python3 -m mypy ${main_dir}