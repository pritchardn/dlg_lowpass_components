"""Python setup.py for dlg_lowpass_components package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("dlg_lowpass_components", "VERSION")
    '0.1.1'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="dlg_lowpass_components",
    version=read("dlg_lowpass_components", "VERSION"),
    description="Awesome dlg_lowpass_components created by pritchardn",
    url="https://github.com/pritchardn/dlg_lowpass_components/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="pritchardn",
    author_email="nicholas.pritchard@uwa.edu.au",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "dlg_lowpass_components = dlg_lowpass_components.__main__:main"
        ]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
