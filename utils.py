import os
import json
import sys


FILES = "./files"



def initialize():
    if not os.path.exists(FILES):
        os.mkdir(FILES)
        cls()

def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def select_file():
    cls()
    files = [file for file in os.listdir(FILES) if file.endswith(".jpg")]
    if len(files) == 0:
        return "file.pdf" if os.path.exists("file.pdf") else None
    print("📁 Select a file")
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")
    print()

    try:
        possible_selections = [i for i in range(len(files) + 1)]
        selection = int(input("Enter a number, or 0 to exit: "))
        if selection == 0:
            handle_exit()
        elif selection not in possible_selections:
            select_file()
        else:
            file_path = os.path.abspath(os.path.join(FILES, files[selection - 1]))
    except ValueError:
        select_file()

    return file_path


def handle_exit():
    print("\nGoodbye!\n")
    exit()
    sys.exit(1)
