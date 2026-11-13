/** Rays per pixel; one bit each in a lookup entry. */
export declare const RAY_COUNT = 32;
/** The spiral both the rays and the depth samples follow. */
export declare const GOLDEN_ANGLE: number;
export interface OccluderLookupTable {
    /** One 32-bit ray mask per entry, x fastest, then y, then radius. */
    readonly data: Uint32Array;
    /** Sphere-mapped directions across. */
    readonly width: number;
    /** Sphere-mapped directions down. */
    readonly height: number;
    /** Radius steps, from 0 to a quarter turn. */
    readonly depth: number;
}
export interface OccluderLookupTableOptions {
    /** Directions along each sphere-map axis. The reference ships 24, 32 and 64. */
    directions?: number;
    /** Radius steps. The reference ships 16, 24 and 32. */
    radii?: number;
}
/** The 32 cosine-distributed ray directions, in tangent space with +z along the normal. */
export declare function rayDirections(): [number, number, number][];
/**
 * Builds the table on the CPU. The default 24x24x16 size takes a few
 * milliseconds and matches the reference's `24x24x16x32_R32_UINT.bin`.
 */
export declare function createOccluderLookupTable({ directions, radii }?: OccluderLookupTableOptions): OccluderLookupTable;
