# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/denehoffman/ganesh/compare/v0.1.0...v0.2.0) - 2024-07-14

### Fixed
- re-implement args that were lost in merge
- move main traits to core module and modify gradient and hessian methods to work better at larger values

### Other
- update docstrings to reflect arguments
- Merge branch 'development' of https://github.com/denehoffman/ganesh into development
- add benchmark
- release

## [0.1.0](https://github.com/denehoffman/ganesh/releases/tag/v0.1.0) - 2024-07-13

### Added
- switch to typed-builder for Result-less builder patterns
- first commit, basic traits, some test functions, and some basic optimization algorithms

### Fixed
- ignore doctests that aren't meant to be tests

### Other
- add metadata to Cargo.toml
- create release.yml
- fix \geq symbol
- update documentation to match README and newer features
- Create LICENSE
- add workflows
- many interconnected changes that I don't care to write individual commits for
- remove BFGS algorithm (for now)
- add KaTeX header
- create README.md
