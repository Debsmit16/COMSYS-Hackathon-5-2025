from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")
requirements = Path("requirements.txt").read_text().splitlines()

setup(
    name="comsys-hackathon-2025",
    version="1.0.0",
    author="Your Name",
    description="Robust face recognition & gender classification (COMSYS-2025)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "comsys-train=scripts.train_face_matcher:main",
            "comsys-evaluate=scripts.evaluate_model:main",
        ]
    },
)