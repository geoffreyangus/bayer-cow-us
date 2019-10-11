"""
"""

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    entry_points={
        'console_scripts': [
            'run=cow_tus.run:run',
            'connect=cow_tus.run:connect'
        ]
    },
    name="cow-tus",
    version="0.0.1",
    author="Geoffrey Angus",
    author_email="gangus@stanford.edu",
    description="Deep Learning on Cow Thoracic Ultrasonography.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seyuboglu/fdg-pet-ct",
    packages=setuptools.find_packages(include=['pet_ct']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch', 'torchvision', 'numpy', 'pandas', 'scipy', 'scikit-learn',
        'click', 'tqdm', 'opencv-python', 'matplotlib', 'seaborn', 'plotly',
    ]
)