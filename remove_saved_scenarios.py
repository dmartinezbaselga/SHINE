import os

paths = ["bad_scenarios", "good_scenarios", "both_wrong", "both_right"]
for path in paths:
    for file_name in os.listdir(path):
        # construct full file path
        file = path + "/" + file_name
        if os.path.isfile(file):
            print('Deleting file:', file)
            os.remove(file)