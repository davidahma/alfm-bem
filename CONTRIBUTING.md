# Contributing

Start with a minimal case that shows the behavior you want to change. Include the Python version, input dimensions, and a seed for numerical examples. For a change to a benchmark, describe what each policy observes, when it observes it, and which denominator you report.

1. Clone the repository, create a branch, and install the research dependencies using the README.
2. Run `make test` and `make reproduce`. For manuscript edits, run `make paper-container` and include the rebuilt release PDF.
3. Open a pull request with the commands you ran and the observable effect. If a result changes, update its JSON, manuscript interpretation, and regression expectation together.

The default branch requires pull requests and passing release checks. Use [GitHub issues](https://github.com/davidahma/alfm-bem/issues) for reproducibility reports and proposed experiments. Report vulnerabilities through the [security policy](SECURITY.md).
