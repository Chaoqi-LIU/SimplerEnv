from pathlib import Path

from setuptools import find_packages, setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="praxis-simpler",
    version="0.0.1",
    author="Xuanlin Li",
    maintainer="Chaoqi Liu",
    maintainer_email="liuchaoqi730@gmail.com",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Chaoqi-LIU/SimplerEnv",
    project_urls={
        "Source": "https://github.com/Chaoqi-LIU/SimplerEnv",
        "Maintainer Website": "https://chaoqi-liu.com",
    },
    packages=find_packages(include=["simpler_env*"]),
    python_requires=">=3.10",
    install_requires=[
        "dm-tree",
        "tyro>=0.8.5",
    ],
)
