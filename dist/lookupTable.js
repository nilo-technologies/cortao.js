// The occluder lookup table: which of CORTAO's 32 rays a circular occluder
// blocks, by the occluder's tangent-space direction and its angular radius.
//
// A port of OccluderLookupTableGenerator.py from the CORTAO reference
// implementation v1.0.1 by Daniel Isheden, https://github.com/theagentd/CORTAO,
// MIT License (see LICENSE-cortao).
/** Rays per pixel; one bit each in a lookup entry. */
export const RAY_COUNT = 32;
/** The spiral both the rays and the depth samples follow. */
export const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
/** The 32 cosine-distributed ray directions, in tangent space with +z along the normal. */
export function rayDirections() {
    const rays = [];
    for (let ray = 0; ray < RAY_COUNT; ray++) {
        const angle = ray * GOLDEN_ANGLE;
        const radius = Math.sqrt((ray + 0.5) / RAY_COUNT);
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        rays.push([x, y, Math.sqrt(1 - x * x - y * y)]);
    }
    return rays;
}
/**
 * Builds the table on the CPU. The default 24x24x16 size takes a few
 * milliseconds and matches the reference's `24x24x16x32_R32_UINT.bin`.
 */
export function createOccluderLookupTable({ directions = 24, radii = 16 } = {}) {
    const rays = rayDirections();
    const slice = directions * directions;
    const data = new Uint32Array(slice * radii);
    for (let z = 0; z < radii; z++) {
        const occluderRadius = (Math.SQRT2 * (z + 0.5)) / radii;
        const occluderCosAngle = 1 - occluderRadius * occluderRadius * 0.5;
        const closest = rays.map(() => ({ index: -1, cosAngle: -1 }));
        for (let y = 0; y < directions; y++) {
            for (let x = 0; x < directions; x++) {
                const index = y * directions + x;
                const ex = ((x + 0.5) / directions) * 2 - 1;
                const ey = ((y + 0.5) / directions) * 2 - 1;
                const dz = Math.max(1 - ex * ex - ey * ey, 0);
                const scale = 2 * Math.sqrt(dz);
                const direction = [ex * scale, ey * scale, dz * 2 - 1];
                if (direction[2] === -1)
                    continue;
                let entry = 0;
                rays.forEach((ray, i) => {
                    const cosAngle = direction[0] * ray[0] + direction[1] * ray[1] + direction[2] * ray[2];
                    if (cosAngle > closest[i].cosAngle)
                        closest[i] = { index, cosAngle };
                    if (cosAngle > occluderCosAngle)
                        entry |= 1 << i;
                });
                data[z * slice + index] = entry >>> 0;
            }
        }
        // Every ray is blocked by the occluder pointing closest to it, however small.
        closest.forEach(({ index }, i) => {
            data[z * slice + index] = (data[z * slice + index] | (1 << i)) >>> 0;
        });
    }
    return { data, width: directions, height: directions, depth: radii };
}
