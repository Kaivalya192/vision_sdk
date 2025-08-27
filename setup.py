from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [r.strip() for r in f if r.strip()]

setup(
    name='dexsdk',
    version='0.1.0',
    description='Python SDK for Dexsent pick_and_place vision module',
    author='Dexsent Robotics',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
)
