# Programming Language Identification

## Introduction

Motivated by the amount of code that goes unidentified on the web, I have explored the existing practical methods for algorithmically identifying the programming language of source code.

### Literature

*"Programming Language Identification (PLI)—also referred to as Source Code Classification (SCC) in the literature—is the problem of identifying the programming language in which a given source code file, or just a short code snippet, is written in. PLI is a common preliminary need in automated program comprehension and also a relevant practical problem for both practitioners and researchers, with important applications in programming trend analysis (Chen et al., 2005), mining software repositories (Mockus, 2009; Caneill, Germn & Zacchiroli, 2017), source code indexing, and code search (Gallardo-Valencia & Sim, 2009; Kononenko et al., 2014)."* ([Del Bonifro et al., 2021](https://upsilon.cc/~zack/research/publications/img-based-lang-detection.pdf))


<br/>

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### PREREQUISITES

Download the Rosetta submodule:

```
$ git submodule update --init --recursive
```

you can find all the references for the Rosetta project at [this link](http://www.rosettacode.org/wiki/Rosetta_Code).

#### Apple M1

If you are using the new Apple M1 chip please be sure to have installed `hdf5` by running:
```
$ brew install hdf5
```

### INSTALLING

Create and enable the virtual environment:
```
$ mkdir venv
$ python -m venv venv
$ source venv/bin/activate
```

below you can find all the scripts for installing based on your OS/processor
```
$ make
  >  +------------------------------------------------------+
  >  |         OS         |  Hardware  |    Setup Command   |
  >  +------------------------------------------------------+
  >  |   Windows/Linux    |   - GPU    |  'make setup.CPU'  |
  >  |    Apple macOS     |    + M1    |  'make setup.M1'   |
  >  |    Apple macOS     |    - M1    |  'make setup.CPU'  |
  >  +------------------------------------------------------+
```

for instance if you have MacOS with Intel chip you have to run:
```
$ make setup.CPU
```

<br/>

## Running the tests

You can train/test the program by running:
```
$ make run
```

or alternatively
```
$ python programming_languages_classification/main.py
```

<br/>

## Built With

* [Tensorflow](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Rosetta](https://maven.apache.org/) - Dependency Management


<br/>

## Authors

* Matteo Conti - *author* - [contimatteo](https://github.com/contimatteo)


<br/>

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
