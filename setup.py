from setuptools import setup, find_packages

setup(
    name="weaponTargetAssignment",
    version="0.0.1",
    description="weaponTargetAssignment",
    author="Jeffrey Chan",
    url="https://github.com/jeffreychan177/weaponTargetAssignment",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["numpy", "gymnasium"],
    include_package_data=True,
)