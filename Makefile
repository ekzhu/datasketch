doc:
	pandoc --from=markdown --to=rst --output=docsrc/index.rst README.md
	sphinx-build -b html docsrc doc
