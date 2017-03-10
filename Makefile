test:
	nosetests

doc:
	sphinx-build -a -b html docsrc docs


upload:
	rm -rf ./dist/*
	python setup.py bdist_wheel --universal
	twine upload ./dist/*

install:
	pip install -e .
