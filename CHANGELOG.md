# Changelog

## 0.2.0

- `CortaoNode` accumulates frames with a reprojected, depth-checked history (`temporalBlend`), and its noise changes every frame, so it stops crawling as the camera moves.
- `CortaoNode` draws at `resolutionScale` of the depth texture's size, 0.5 by default, instead of the drawing buffer's, so it follows a scene pass drawn below the canvas size.
- Each pass has its own quad, so a draw no longer sets up its pipeline again.

## 0.1.0

- `CortaoNode` for `WebGPURenderer`, `CortaoPass` for `WebGLRenderer`.
- Occluder lookup table generated at run time, equal to the reference's 24x24x16 table.
- Examples: comparison harness against GTAOPass and a ray-traced reference, WebGPU demo with GPU ms readout.
