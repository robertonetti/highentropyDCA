from setuptools import setup, find_packages

setup(
    name='highentDCA',
    version='0.0',
    author='Roberto Netti, Martin Weigt',
    maintainer='Roberto Netti',
    author_email='robertonetti3@gmail.com',
    description='Python implementation of Direct Coupling Analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/robertonetti/highentropyDCA',
    packages=find_packages(include=['highentDCA', 'highentDCA.*']),
    include_package_data=True,
    python_requires='>=3.10',
    license_files=["LICENSE"],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'highentDCA=highentDCA.cli:main',
        ],
    },
    install_requires=[
        'matplotlib>=3.8.0',
        'numpy>=1.26.4',
        'pandas>=2.2.2',
        'torch>=2.1.0',
        'tqdm>=4.66.6',
        'wandb>=0.12.0',
        'biopython>=1.85',
        'adabmDCA==0.5.0',
    ],
)
