test:
	nosetests --exclude-dir=test/aio

doc:
	sphinx-build -a -b html docsrc docs

upload:
	rm -rf ./dist/*
	python setup.py bdist_wheel --universal
	python setup.py sdist
	twine upload ./dist/*

install:
	pip install -e .
