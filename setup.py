from setuptools import setup, find_packages

setup(
    name="rfcapture",
    version="1.0.0",
    description="Advanced RF Signal Analysis with ML/DL",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.5.0',
        'pyyaml>=6.0',
        'pyrtlsdr>=0.3.0',
        'scikit-learn>=1.0.0',
        'h5py>=3.6.0',
        'tqdm>=4.62.0',
        'joblib>=1.2.0',
    ],
    entry_points={
        'console_scripts': [
            'rfcapture=cli.simple_main:main',
        ],
    },
    python_requires='>=3.8',
)