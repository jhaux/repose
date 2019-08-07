from setuptools import setup, find_packages

setup(
    name="repose",
    version="0.1",
    description="Compare keypoints in the edflow eval setting",
    url="https://github.com/jhaux/repose",
    author="Johannes Haux",
    author_email="johannes.haux@iwr.uni-heidelberg.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "edflow",
        "tqdm",
        "numpy",
    ],
    zip_safe=False,
)
