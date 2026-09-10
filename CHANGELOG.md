# Changelog

## [Unreleased]

### Added

- JSON-backed storage now accepts AMSTensor directly, including strided and
  accelerator-resident tensors, and works in builds without Torch.
- JSON-backed storage can emit binary tensor files in rank-qualified case
  directories or self-contained base64 manifests, with a separate manifest for
  each domain and rank (#205).

### Changed

- Workflow environments can now use active system Flux Python bindings instead
  of installing `flux-python` through default AMS Python dependencies.
- Added `AMS_INSTALL_FLUX_PYTHON` so non-system workflow builds can opt into
  installing the `flux-python` optional dependency through CMake.
