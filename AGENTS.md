# Agents Guide for datasketch

This document provides guidance for AI agents working with the datasketch repository.

## Project Overview

**datasketch** is a Python library that provides probabilistic data structures for processing and searching very large datasets with minimal loss of accuracy. The library is designed for big data applications where traditional exact methods are too slow or memory-intensive.

## Core Components

### Data Sketches
- **MinHash** (`minhash.py`): Estimates Jaccard similarity and cardinality
- **Weighted MinHash** (`weighted_minhash.py`): Estimates weighted Jaccard similarity
- **B-Bit MinHash** (`b_bit_minhash.py`): Memory-efficient variant of MinHash
- **Lean MinHash** (`lean_minhash.py`): Compressed MinHash representation
- **HyperLogLog** (`hyperloglog.py`): Estimates cardinality of large datasets
- **HyperLogLog++** (`hyperloglog.py`): Improved HyperLogLog with better accuracy

### Indexing Structures
- **MinHash LSH** (`lsh.py`): Locality-Sensitive Hashing for Jaccard similarity threshold queries
- **MinHash LSH Forest** (`lshforest.py`): LSH-based index for Jaccard top-K queries
- **MinHash LSH Ensemble** (`lshensemble.py`): LSH for containment threshold queries
- **HNSW** (`hnsw.py`): Hierarchical Navigable Small World graphs for custom metric top-K queries

### Storage Layer
- **Storage** (`storage.py`): Abstraction layer supporting in-memory, Redis, and Cassandra backends

## Development Guidelines

### Project Structure
```
datasketch/
├── datasketch/          # Main package code
│   ├── experimental/    # Experimental features (async implementations)
│   └── *.py            # Core modules
├── test/               # Test suite
├── benchmark/          # Performance benchmarks
├── examples/           # Usage examples
└── docs/              # Sphinx documentation
```

### Technology Stack
- **Language**: Python 3.7+
- **Core Dependencies**: NumPy (>=1.11), SciPy (>=1.0.0)
- **Optional Dependencies**: 
  - Redis (>=2.10.0) for distributed storage
  - Cassandra driver (>=3.20) for Cassandra storage
  - Various testing and benchmarking tools

### Testing
- **Framework**: pytest
- **Command**: `pytest` from repository root
- **CI/CD**: GitHub Actions for Python 3.7-3.11
- Tests are located in the `test/` directory
- Specialized tests exist for Cassandra and MongoDB integration

### Code Style
- **Linter**: flake8
- Configuration in `.flake8` file
- Run: `flake8` from repository root

### Building Documentation
- **Tool**: Sphinx
- **Location**: `docs/` directory
- Built automatically via GitHub Actions on push

## Key Concepts

### Probabilistic Data Structures
This library implements algorithms that trade perfect accuracy for significant improvements in speed and memory usage. Understanding the probabilistic nature and error bounds is crucial when working with this code.

### Jaccard Similarity
Many components deal with Jaccard similarity: the intersection over union of two sets. This is a fundamental metric for comparing documents, user behaviors, and other set-based data.

### LSH (Locality-Sensitive Hashing)
The indexing structures use LSH to enable fast approximate nearest neighbor search. Changes to LSH components should preserve the probabilistic guarantees.

## Common Tasks

### Adding a New Data Sketch
1. Create a new Python module in `datasketch/`
2. Implement the data structure with clear docstrings
3. Add comprehensive tests in `test/`
4. Add documentation in `docs/`
5. Update `__init__.py` to export the new class
6. Add usage examples if appropriate

### Modifying Storage Layer
- Storage implementations must follow the interface defined in `storage.py`
- Test with both in-memory and external storage backends
- Be mindful of serialization/deserialization performance

### Performance Considerations
- Many operations are performance-critical
- Benchmark significant changes using tools in `benchmark/`
- NumPy operations should be vectorized when possible
- Avoid Python loops over large datasets

## Testing Strategy

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test storage backends and complex workflows
- **Benchmark tests**: Verify performance characteristics
- **Cassandra/MongoDB tests**: Run via specialized GitHub Actions workflows

## External Resources

- **Documentation**: https://ekzhu.github.io/datasketch
- **GitHub**: https://github.com/ekzhu/datasketch
- **PyPI**: https://pypi.org/project/datasketch/

## Notes for AI Agents

- When modifying core algorithms, preserve mathematical properties and probabilistic guarantees
- Always run tests after changes: `pytest`
- Check code style: `flake8`
- The codebase values performance - profile before making optimization claims
- Documentation is in reStructuredText format (.rst files)
- Many modules have academic paper references in docstrings - these are important context
