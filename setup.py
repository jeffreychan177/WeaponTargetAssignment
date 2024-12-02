from setuptools import setup, find_packages

setup(
    name="WeaponTargetAssignment",
    version="0.0.1",
    description="WeaponTargetAssignment",
    author="Jeffrey Chan",
    url="https://github.com/jeffreychan177/WeaponTargetAssignment",
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