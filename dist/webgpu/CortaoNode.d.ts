import type { NodeBuilder, NodeFrame, TextureNode } from 'three/webgpu';
import * as THREE from 'three/webgpu';
/** Defaults from the reference's `CORTAOConfigurationParameters`. */
export declare const CORTAO_DEFAULTS: {
    readonly maxScreenspaceRadiusHeightRatio: 0.2;
    readonly maxSlopeCompensation: 4;
    readonly backfaceThreshold: 0.5;
    readonly backfaceBoost: 0.5;
    readonly filterDepthSensitivity: 512;
    readonly filterNormalSensitivity: 16;
};
/**
 * Ambient occlusion from a scene pass's depth, as a texture whose red channel
 * is 1 where nothing is occluded. It draws while `enabled`, in three passes
 * before the frame: pack, occlusion, filter. Perspective cameras only.
 *
 * ```ts
 * const scenePass = pass(scene, camera);
 * const cortao = new CortaoNode(scenePass.getTextureNode('depth'), () => camera);
 * const ao = cortao.getTextureNode().r;
 * pipeline.outputNode = vec4(scenePass.getTextureNode().rgb.mul(ao), 1);
 * ```
 */
export declare class CortaoNode extends THREE.TempNode<'float'> {
    private readonly depthNode;
    private readonly camera;
    static get type(): string;
    /** While false nothing is drawn and the texture keeps its last contents. */
    enabled: boolean;
    /** Depth samples per pixel; a change rebuilds the occlusion shader. */
    sampleCount: number;
    /** Frames seen since creation, whether or not occlusion was drawn. */
    frames: number;
    /** The occlusion radius, in world units. */
    readonly radius: THREE.UniformNode<"float", number>;
    private readonly packedTarget;
    private readonly occlusionTarget;
    private readonly filteredTarget;
    private readonly lookupTable;
    private readonly packMaterial;
    private readonly occlusionMaterial;
    private readonly filterMaterial;
    private readonly textureNode;
    private readonly projection;
    private readonly projectionInverse;
    private readonly near;
    private readonly far;
    /** Texture coordinates and linear depth to view space: xy scale, zw offset. */
    private readonly unprojectParams;
    private readonly normalBias;
    private readonly maxApparentCircleRadiusCoefficient;
    private builtSampleCount;
    private buildOcclusion;
    private sharedContext;
    /**
     * @param depthNode A scene pass's depth texture.
     * @param camera Read every frame, so a pass that changes cameras is followed.
     */
    constructor(depthNode: TextureNode, camera: () => THREE.Camera);
    /** The filtered occlusion. */
    getTextureNode(): TextureNode;
    private viewPosition;
    /** The view-space normal rebuilt from depth; the scene pass has no normal buffer. */
    private normalFromDepth;
    /** Linear depth and normal packed into the red channel; 0 on the background. */
    private pack;
    /** The reference's main pass, with the sample loop unrolled. */
    private occlusion;
    /** The reference's sparse filter: neighbors weighted by how well they fit the pixel's depth plane and normal. */
    private filter;
    private setSize;
    updateBefore(frame: NodeFrame): undefined;
    setup(builder: NodeBuilder): TextureNode;
    dispose(): void;
}
