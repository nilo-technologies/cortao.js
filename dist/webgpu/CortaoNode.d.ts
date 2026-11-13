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
    /** The occlusion's size as a fraction of the depth texture's; 0.5 is a quarter of the pixels. */
    resolutionScale: number;
    /**
     * How much of each new frame goes into the history, 1 for none kept. Lower is
     * smoother and slower to follow moving objects; history the camera moved away
     * from, by depth, is dropped.
     */
    readonly temporalBlend: THREE.UniformNode<"float", number>;
    private readonly packedTarget;
    private readonly occlusionTarget;
    private readonly filteredTarget;
    /**
     * Accumulated occlusion in red and its linear depth in green, in turns: each
     * frame writes one from the other, and the texture node reads the one written.
     */
    private readonly resolvedTargets;
    private readonly lookupTable;
    private readonly packMaterial;
    private readonly occlusionMaterial;
    private readonly filterMaterial;
    /** One per resolved target it writes, reading the other. */
    private readonly temporalMaterials;
    private readonly textureNode;
    /** One quad per pass: a quad whose material changes every draw is set up again every draw. */
    private readonly quads;
    /** Which resolved target this frame writes. */
    private parity;
    private readonly projection;
    private readonly projectionInverse;
    private readonly near;
    private readonly far;
    /** Texture coordinates and linear depth to view space: xy scale, zw offset. */
    private readonly unprojectParams;
    private readonly normalBias;
    private readonly maxApparentCircleRadiusCoefficient;
    /** Shifts the per-pixel noise every frame, so the history averages different samples. */
    private readonly noiseOffset;
    private readonly cameraWorld;
    private readonly previousViewProjection;
    /** 0 when the history is from another size or there is none yet. */
    private readonly historyValid;
    private readonly targetSize;
    private builtSampleCount;
    private buildOcclusion;
    private sharedContext;
    /**
     * @param depthNode A scene pass's depth texture.
     * @param camera Read every frame, so a pass that changes cameras is followed.
     */
    constructor(depthNode: TextureNode, camera: () => THREE.Camera);
    /** The filtered, accumulated occlusion in red. */
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
    /**
     * The filtered occlusion blended into last frame's, found where this pixel's
     * surface was then. The history is dropped where its depth does not match
     * that place, off screen, and on the background.
     */
    private temporal;
    private setSize;
    updateBefore(frame: NodeFrame): undefined;
    setup(builder: NodeBuilder): TextureNode;
    dispose(): void;
}
