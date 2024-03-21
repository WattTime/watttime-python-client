from setuptools import setup

setup(
    name="watttime",
    description="An officially maintained python client for WattTime's API providing access to electricity grid emissions data.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    version="v1.2",
    packages=["watttime"],
    python_requires=">=3.8",
    install_requires=["requests", "pandas>1.0.0", "python-dateutil"],
)
