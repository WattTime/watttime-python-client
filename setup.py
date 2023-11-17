from setuptools import setup

setup(
    name="watttime_sdk",
    description="""
        A python software development kit with basic examples for using the
        WattTime API, including data that spans across the 30 day API request limit.""",
    version="3.0",
    packages=["watttime_sdk"],
    python_requires=">=3.7",
    install_requires=[
        "requests",
        "pandas>1.0.0"
    ]
)