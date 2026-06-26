from setuptools import setup, find_packages

setup(
    name='vecdb',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['requests>=2.28.1'],
    python_requires='>=3.9',
    description='Python client for the vecdb vector database',
)