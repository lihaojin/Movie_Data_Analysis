imUsingColab = False
#
# if imUsingColab:
#     !pip install python-dotenv
#     !pip install kaggle

import zipfile
import subprocess
import os.path
from dotenv import load_dotenv
load_dotenv()

def download(imUsingColab=False):
    zipPath = '/content/the-movies-dataset.zip' if imUsingColab else 'the-movies-dataset.zip'
    zipExists = os.path.isfile(zipPath)

    if not zipExists:
        print("Couldn't Find dataset downloading now")
        print(subprocess.check_output(['kaggle', 'datasets', 'download', '-d', 'rounakbanik/the-movies-dataset']))

    print("Opening zip file")
    zip_ref = zipfile.ZipFile(zipPath, 'r')
    print(zip_ref.namelist())
    print("Loaded Zip File")
    print("Attempting extraction")
    zip_ref.extractall('./data')
    print("Extraction Completed")
    zip_ref.close()
