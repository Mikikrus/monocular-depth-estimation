import argparse
import re


def replace_string_in_file(path: str, old_string: str, new_string: str) -> None:
    with open(path, mode="r") as f:
        file_contents = f.read()
    file_contents = re.sub(old_string, new_string, file_contents)
    with open(path, mode="w") as f:
        f.write(file_contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to file")
    parser.add_argument("--old_string", help="string to replace")
    parser.add_argument("--new_string", help="string to subtitude with")

    args = parser.parse_args()
    replace_string_in_file(args.path, args.old_string, args.new_string)
