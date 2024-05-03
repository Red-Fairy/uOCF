import os

root = '/viscam/projects/uorf-extension/uOCF/scripts'

# iterate over all files in this directory and find one with target word "bridge"

for subdir, dirs, files in os.walk(root):
    for file in files:
        with open(os.path.join(subdir, file), 'r') as f:
            if 'bridge' in f.read():
                print(os.path.join(subdir, file))