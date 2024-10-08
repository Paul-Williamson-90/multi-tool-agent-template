test:
	PYTHONPATH=. pytest -vv --failed-first --cov=./src/ --cov-report=term-missing --cov-report=json -s -p no:warnings --cov-fail-under=100 --junitxml=test-results/pytest_report.xml