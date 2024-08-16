"""Setup.py for OpenEA-torch."""

import os

import setuptools

MODULE = 'openea_torch'
VERSION = '1.0'
PACKAGES = setuptools.find_packages(where='src')
META_PATH = os.path.join('src', MODULE, '__init__.py')
KEYWORDS = ['Knowledge Graph', 'Embeddings', 'Entity Alignment']
INSTALL_REQUIRES = [
    # 'tensorflow',
    'torch',
    'pandas',
    'matching',
    'scikit-learn',
    'numpy',
    'gensim',
    'python-Levenshtein',
    'scipy',
]

if __name__ == '__main__':
    setuptools.setup(
        name=MODULE,
        version=VERSION,
        description='A package for embedding-based entity alignment',
        url='',
        author='Noo',
        author_email='',
        maintainer='',
        maintainer_email='',
        license='MIT',
        keywords=KEYWORDS,
        packages=setuptools.find_packages(where='src'),
        package_dir={'': 'src'},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        zip_safe=False,
    )
# graph-tool安不上