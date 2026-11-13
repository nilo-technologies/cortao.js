# Agent Instructions

cortao.js is Nilo's three.js port of CORTAO, `theagentd/CORTAO` (credited in `LICENSE-cortao`).

## Pull requests

- All PRs go to `nilo-technologies/cortao.js`, base `main`. Never open PRs, issues or comments on `theagentd/CORTAO`.
- Pass the repo explicitly: `gh pr create --repo nilo-technologies/cortao.js --base main --head <branch>`.

## Layout

- `src/lookupTable.ts`: occluder lookup table generator (port of `OccluderLookupTableGenerator.py`).
- `src/webgpu/CortaoNode.ts`: TSL node for `WebGPURenderer`.
- `src/webgl/CortaoPass.js`: WebGL2 pass for `WebGLRenderer`.
- `dist/`: built output, committed so `github:` installs work without a build. Rebuild after changing `src/`.
- `examples/`: Vite workspace package `@cortao/examples` (comparison harness, WebGPU demo, Playwright scripts).
- `test/lookup-table.mjs`: generator against the reference table in `test/fixtures/`.

## Commands

```bash
pnpm build       # tsc into dist/
pnpm typecheck
pnpm test        # build, then the lookup table test
pnpm check       # biome lint and format check
pnpm dev         # build, then the examples on http://127.0.0.1:4791/
```

## Rules

- Report performance in milliseconds, never frames per second. Say whether a time is GPU (timer or timestamp queries) or wall clock.
- The ray-traced reference and video capture keep the GPU busy for minutes; do not start them unasked on a shared machine.
- Do not commit Sponza or other third-party models; `pnpm fetch-sponza` downloads the geometry.
