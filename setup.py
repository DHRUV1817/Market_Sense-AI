
from setuptools import setup, find_packages

setup(
    name="marketsense",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.31.1",
        "flask>=2.3.3",
        "pandas>=2.2.0",
        "numpy>=1.26.2",
        "groq>=0.4.2",
        "python-dotenv>=1.0.0",
        "plotly>=5.18.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.2",
        "requests>=2.31.0",
        "scipy>=1.12.0",
        "transformers>=4.37.0",
        "torch>=2.1.0",
        "nltk>=3.8.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.12.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ]
    },
    python_requires=">=3.7",
)
