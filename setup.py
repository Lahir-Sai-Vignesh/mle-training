from setuptools import find_packages, setup

setup(
    name="my_package",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "six",
        "matplotlib",
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            "run-script=my_package.nonstandardcode:main",
        ],
    },
)
