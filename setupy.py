import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='benchmark_ml_models',
    version='0.0.1',
    author="Carlos Donado",
    author_email='cf.donado@hotmail.com',
    long_description=long_description,
    long_description_content_type='test/markdown',
    url="https://github.com/carlosdonado10/GeneralPurposePredictor.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'sklearn',
        'pandas'
    ]
)
