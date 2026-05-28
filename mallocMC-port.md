# mallocMC alpaka3 port

## Plan

- [x] Inspect `mallocMC` for legacy alpaka integration points and identify likely alpaka3 replacements.
- [ ] Update dependency fetching and CMake integration to pull `alpaka-group/alpaka3` and match alpaka3's build requirements.
- [ ] Replace legacy accelerator/tag/device/queue usage in library headers with alpaka3 APIs.
- [ ] Replace legacy accelerator/tag/device/queue usage in tests and examples with alpaka3 APIs.
- [ ] Update warp-size handling to use alpaka3 `onAcc::Acc` compile-time information where required.
- [ ] Build and run the CPU test suite.
- [ ] Configure and compile the project with `nvcc` without running GPU tests.
- [ ] Do a final regression pass and record remaining risks.

## Notes

- Keep changes local to `mallocMC`.
- Prefer API adapters or small helper traits over wide rewrites where possible.
