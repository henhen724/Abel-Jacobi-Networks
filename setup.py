from setuptools import setup, find_packages

setup(
    packages=find_packages(where="src", include=["aj", "aj.*"]),
    package_dir={"": "src"},
)
