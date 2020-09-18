from io import open
import os
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install
import pt_mlagents.trainers

VERSION = pt_mlagents.trainers.__version__
EXPECTED_TAG = pt_mlagents.trainers.__release_tag__

here = os.path.abspath(os.path.dirname(__file__))


class VerifyVersionCommand(install):
    """
    Custom command to verify that the git tag is the expected one for the release.
    Based on https://circleci.com/blog/continuously-deploying-python-packages-to-pypi-with-circleci/
    This differs slightly because our tags and versions are different.
    """

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag != EXPECTED_TAG:
            info = "Git tag: {0} does not match the expected tag of this app: {1}".format(
                tag, EXPECTED_TAG
            )
            sys.exit(info)


# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pt_mlagents",
    version=VERSION,
    description="Unity Machine Learning Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Unity-Technologies/ml-agents",
    author="Mengqi Ye",
    author_email="sevenseaswander@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    # find_namespace_packages will recurse through the directories and find all the packages
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    zip_safe=False,
    install_requires=[
        # Test-only dependencies should go in test_requirements.txt, not here.
        # "grpcio>=1.11.0",
        # "h5py>=2.9.0",
        # "mlagents_envs=={}".format(VERSION),
        # "numpy>=1.13.3,<2.0",
        # "Pillow>=4.2.1",
        # "protobuf>=3.6",
        # "pyyaml>=3.1.0",
        # "tensorflow-gpu>=1.7,<3.0",
        # "cattrs>=1.0.0",
        # "attrs>=19.3.0",
        # 'pypiwin32==223;platform_system=="Windows"',
        # # We don't actually need six, but tensorflow does, and pip seems
        # # to get confused and install the wrong version.
        # "six>=1.12.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pt_mlagents-learn=pt_mlagents.trainers.learn:main",
            "pt_mlagents-run-experiment=pt_mlagents.trainers.run_experiment:main",
        ]
    },
    cmdclass={"verify": VerifyVersionCommand},
)
