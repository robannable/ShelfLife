from setuptools import setup, find_packages

setup(
    name="shelflife",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.8",
    description="An intelligent library cataloguing tool that transforms minimal book input into rich, interconnected bibliographic data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 