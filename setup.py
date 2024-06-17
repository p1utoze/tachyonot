from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read()

setup(
    name='simatic',
    version='0.1.0',
    python_requires='>=3.9',
    packages=find_packages(),
    py_modules=["simatic"],
    install_requires=[
        requirements
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cpu"
    ],
    entry_points={
        'console_scripts': [
            'simatic-cli=simatic.inference:llm_inference',
        ],
    },
)