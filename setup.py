from setuptools import setup

setup(
    name="watttime",
    description="An officially maintained python client for WattTime's API providing access to electricity grid emissions data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version="v1.3.3",
    packages=["watttime", "watttime.evaluation"],
    python_requires=">=3.8",
    install_requires=["requests", "pandas>1.0.0", "holidays", "python-dateutil"],
    extras_require={"report": ["papermill", "nbconvert", "plotly", "scipy"]},
    scripts=["watttime/report.py"],
)
