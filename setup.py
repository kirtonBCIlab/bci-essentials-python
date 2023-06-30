from setuptools import setup, find_packages

setup(
    name="bci-essentials-python",
    version="0.0.1",
    description="Python backend for bci-essentials",
    author="Brian Irvine",
    packages=find_packages(),
    platforms="any",
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "joblib",
        "pandas",
        "pylsl",
        "pyxdf",
        "matplotlib",
        "seaborn",
        "wheel",
        "pyriemann",
        "tensorflow; platform_machine != 'arm64' ",
        "tensorflow-macos; platform_machine == 'arm64' ",
    ],
)
