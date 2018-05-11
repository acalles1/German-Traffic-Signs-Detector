'''
Title           :app.py
Description     :Click app that solves Kiwi's deep-learning challenge.
Author          :Alejandro Calle-Saldarriaga.
Date Created    :11/05/18
Date Modified   :
version         :0.1
usage           :
input           :
output          :
python_version  :2.7.13
'''"""
M
"""

import click
import os
import requests

@click.group()
def cli():
    pass

@click.command()
def download():
    """
    Downloads dataset, unzips it and saves it under the correct folders.
    """
    localFilePath = './images/dataset.zip'
    url = 'http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'
    r = requests.get(url, allow_redirects=True)
    open(localFilePath, 'wb').write(r.content)
    os.system('unzip images/dataset.zip')
    os.system('rm images/dataset.zip')

cli.add_command(download)

if __name__ == '__main__':
    cli()
