from setuptools import setup

setup(
    name="watttime",
    description="""
        A python software development kit with basic examples for using the
        WattTime API, including data that spans across the 30 day API request limit.""",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    version="v1.0",
    packages=["watttime"],
    python_requires=">=3.8",
    install_requires=["requests", "pandas>1.0.0", "python-dateutil"],
)
