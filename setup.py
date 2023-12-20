from setuptools import setup

setup(
    name="watttime_client",
    description="""
        A python software development kit with basic examples for using the
        WattTime API, including data that spans across the 30 day API request limit.""",
    version="3.0",
    packages=["watttime_client"],
    python_requires=">=3.8",
    install_requires=["requests", "pandas>1.0.0", "python-dateutil"],
)
