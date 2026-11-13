/**
 * The occluder lookup table as an R32UI 3D texture, built on the CPU.
 * @param {{ directions?: number, radii?: number }} [options]
 */
export function createOccluderLookupTexture(options?: {
    directions?: number;
    radii?: number;
}): THREE.Data3DTexture;
/**
 * Loads one of the reference repo's prebuilt 32-ray R32_UINT tables (`tables/occluder`).
 * `createOccluderLookupTexture` builds the same data without a download.
 * @param {string} url
 * @param {'24x24x16' | '32x32x24' | '64x64x32'} [size]
 */
export function loadOccluderLookupTable(url: string, size?: "24x24x16" | "32x32x24" | "64x64x32"): Promise<THREE.Data3DTexture>;
export const VERTEX: "\nin vec3 position;\nvoid main() {\n\tgl_Position = vec4(position.xy, 0.0, 1.0);\n}";
export namespace CORTAO_DEFAULTS {
    let sampleCount: number;
    let maxRadius: number;
    let maxScreenspaceRadiusHeightRatio: number;
    let mipBias: number;
    let maxSlopeCompensation: number;
    let backfaceThreshold: number;
    let backfaceBoost: number;
    let filterDepthSensitivity: number;
    let filterNormalSensitivity: number;
    let largeFilter: boolean;
    let filterSymmetry: boolean;
}
/**
 * Runs CORTAO from a depth texture and a view-space normal texture
 * (rgb = n * 0.5 + 0.5, as MeshNormalMaterial writes it).
 */
export class CortaoPass {
    /**
     * @param {number} width
     * @param {number} height
     * @param {THREE.Data3DTexture | null} [occluderLookupTable] 32-ray R32UI table; built with `createOccluderLookupTexture` when left out, or null to trace the rays in the shader.
     */
    constructor(width: number, height: number, occluderLookupTable?: THREE.Data3DTexture | null);
    width: number;
    height: number;
    params: {
        sampleCount: number;
        maxRadius: number;
        maxScreenspaceRadiusHeightRatio: number;
        mipBias: number;
        maxSlopeCompensation: number;
        backfaceThreshold: number;
        backfaceBoost: number;
        filterDepthSensitivity: number;
        filterNormalSensitivity: number;
        largeFilter: boolean;
        filterSymmetry: boolean;
    };
    packedTarget: THREE.WebGLRenderTarget<THREE.Texture<unknown, THREE.TextureEventMap>>;
    occlusionTarget: THREE.WebGLRenderTarget<THREE.Texture<unknown, THREE.TextureEventMap>>;
    filteredTarget: THREE.WebGLRenderTarget<THREE.Texture<unknown, THREE.TextureEventMap>>;
    prepassMaterial: THREE.RawShaderMaterial;
    mainMaterial: THREE.RawShaderMaterial;
    filterMaterial: THREE.RawShaderMaterial;
    quad: FullScreenQuad;
    setDefine(material: any, name: any, value: any): void;
    /**
     * populatePrepassParameters, populateMainPassParameters and populateFilterPassParameters.
     * @param {THREE.PerspectiveCamera} camera
     * @param {THREE.DepthTexture} depthTexture
     * @param {THREE.Texture} normalTexture
     * @param {number} randomSeed
     */
    updateUniforms(camera: THREE.PerspectiveCamera, depthTexture: THREE.DepthTexture, normalTexture: THREE.Texture, randomSeed: number): void;
    renderMaterial(renderer: any, material: any, target: any): void;
    /**
     * Renders occlusion into `occlusionTarget` (unfiltered) and, when `filter`
     * is set, into `filteredTarget`. Returns the texture holding the result.
     * Use a new `randomSeed` every frame when a temporal filter follows.
     * @param {THREE.WebGLRenderer} renderer
     * @param {THREE.PerspectiveCamera} camera
     * @param {THREE.DepthTexture} depthTexture
     * @param {THREE.Texture} normalTexture View-space normals, rgb = n * 0.5 + 0.5.
     * @param {{ randomSeed?: number, filter?: boolean }} [options]
     * @returns {THREE.Texture}
     */
    render(renderer: THREE.WebGLRenderer, camera: THREE.PerspectiveCamera, depthTexture: THREE.DepthTexture, normalTexture: THREE.Texture, { randomSeed, filter }?: {
        randomSeed?: number;
        filter?: boolean;
    }): THREE.Texture;
    dispose(): void;
}
import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
