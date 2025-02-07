from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req_file:
        return req_file.read().splitlines()

setup(
    name='AutoSpectraEngine',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements(),
)