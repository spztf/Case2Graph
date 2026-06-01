# Contributing

Thank you for your interest in contributing to this project!

## Ways to Contribute

- **Bug reports:** Open an issue describing the problem and steps to reproduce
- **Feature requests:** Open an issue with a clear use case
- **Code contributions:** Fork → branch → PR with description
- **Documentation:** Improvements to README, docstrings, or docs/
- **Reproducibility:** Verify results on different hardware/software versions

## Development Setup

```bash
git clone https://github.com/spztf/Case2Graph.git
cd Case2Graph
conda create -n tax-retrieval python=3.10 -y
conda activate tax-retrieval
pip install -r requirements.txt

# Extract case graphs
tar xzf data/case_graphs.tar.gz -C data/
```

## Code Style

- Python 3.10+ with type hints where practical
- Follow PEP 8
- Use descriptive variable names
- Add docstrings for public functions

## Pull Request Process

1. Ensure code runs without errors
2. Update documentation if needed
3. Describe what changed and why
4. One PR per logical change

## Data Access

The tax case data is provided for academic research only. If you need access to the full enterprise subgraphs (24 GB), please contact the corresponding author with your institutional affiliation and research purpose.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
