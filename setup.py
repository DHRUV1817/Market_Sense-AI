
from setuptools import setup, find_packages

setup(
    name="marketsense",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "flask",
        "pandas",
        "numpy",
        "openai",
        "python-dotenv",
        "plotly",
        "scikit-learn",
        "matplotlib",
        "requests",
        "scipy",
    ],
    python_requires=">=3.7",
)
