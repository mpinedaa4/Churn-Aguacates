from setuptools import setup, find_packages

setup(
    name="extremality_mkl",
    version="0.1.0",
    description="Combinación de kernels usando ordenamiento por extremalidad",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pandas",
    ],
)
