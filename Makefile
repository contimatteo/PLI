# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Signifies our desired python version
# Makefile macros (or variables) are defined a little bit differently than traditional bash, keep 
# in mind that in the Makefile there's top-level Makefile-only syntax, and everything else is bash 
# script syntax.
PYTHON = python
PIP = ${PYTHON} -m pip

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.DEFAULT_GOAL = help

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "+------------------------------------------------------+"
	@echo "|         OS         |  Hardware  |    Setup Command   |"
	@echo "+------------------------------------------------------+"
	@echo "|   Windows/Linux    |   - GPU    |  'make setup.CPU'  |"
	@echo "|    Apple macOS     |    + M1    |  'make setup.M1'   |"
	@echo "|    Apple macOS     |    - M1    |  'make setup.CPU'  |"
	@echo "+------------------------------------------------------+"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

setup.CPU:
	${PIP} install -r ./tools/requirements/CPU.txt
	./tools/scripts/M1.install.zsh
setup.M1:
	./tools/scripts/M1.install.zsh

run:
	${PYTHON} programming_languages_classification/main.py

clean:
	rm -rf __pycache__

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY: help
.PHONY: setup.M1
.PHONY: setup.CPU
.PHONY: run
.PHONY: clean
