from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="msr",
    version="0.1.0",
    author="Darwin Labs",
    author_email="info@darwin-labs.com",
    description="A framework to enhance reasoning capabilities of foundation models through structured multi-step thinking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darwin-labs/MSR",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
) 