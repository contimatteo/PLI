from setuptools import setup


with open("README.md", 'r') as f:
    long_description = f.read()


setup(
    name='plc-problem-solver',
    version='1.0',
    description='Programming Languages Classification solver.',
    # license="MIT",
    long_description = long_description,
    author='Matteo Conti',
    # author_email='?',
    # url="http://www.foopackage.com/",
    packages=['dataset-loader', 'tokenizer', 'solver'],
    # scripts=['scripts/cool']
)
