// CORTAO (Circular Occluder Ray-Traced Ambient Occlusion) as a TSL node.
//
// A port of the CORTAO reference implementation v1.0.1 by Daniel Isheden,
// https://github.com/theagentd/CORTAO, MIT License, Copyright (c) 2026 Daniel
// Isheden. The reference's notice: "Permission is hereby granted, free of
// charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so, subject to
// the following conditions: The above copyright notice and this permission
// notice shall be included in all copies or substantial portions of the
// Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND."
//
// Where this differs from the reference:
// - It reads only a depth texture; normals are rebuilt from depth.
// - The packed buffer has the reference's layout, linear depth and a 10-bit
//   normal in one 32-bit value, but no mip chain; every sample reads the
//   full-size texture.
// - Per-pixel random numbers come from TSL's noise instead of the integer hash.
// - It draws at a fraction of the depth texture's size (`resolutionScale`) and
//   accumulates frames with a reprojected history (`temporalBlend`), so the
//   per-pixel noise, which changes every frame, averages out as the camera moves.

import * as TSL from 'three/tsl';
import type { Node, NodeBuilder, NodeFrame, TextureNode } from 'three/webgpu';
import * as THREE from 'three/webgpu';
import { createOccluderLookupTable, GOLDEN_ANGLE, RAY_COUNT } from '../lookupTable.js';

/** Defaults from the reference's `CORTAOConfigurationParameters`. */
export const CORTAO_DEFAULTS = {
	maxScreenspaceRadiusHeightRatio: 0.2,
	maxSlopeCompensation: 4,
	backfaceThreshold: 0.5,
	backfaceBoost: 0.5,
	filterDepthSensitivity: 512,
	filterNormalSensitivity: 16,
} as const;
const DEFAULTS = CORTAO_DEFAULTS;

/** The large filter's offsets; each is also sampled mirrored. */
const FILTER_OFFSETS: readonly (readonly [number, number])[] = [
	[3, 0],
	[6, 0],
	[3, 3],
	[0, 3],
	[0, 6],
	[3, -3],
];

/**
 * The occluder lookup table, with each 32-bit entry stored as four bytes so it
 * can be sampled as a plain 8-bit texture on every backend.
 */
function createOccluderLookupTexture(): THREE.Data3DTexture {
	const { data, width, height, depth } = createOccluderLookupTable();
	const table = new THREE.Data3DTexture(new Uint8Array(data.buffer), width, height, depth);
	table.format = THREE.RGBAFormat;
	table.type = THREE.UnsignedByteType;
	table.minFilter = THREE.NearestFilter;
	table.magFilter = THREE.NearestFilter;
	table.wrapS = THREE.ClampToEdgeWrapping;
	table.wrapT = THREE.ClampToEdgeWrapping;
	table.wrapR = THREE.ClampToEdgeWrapping;
	table.colorSpace = THREE.NoColorSpace;
	table.generateMipmaps = false;
	table.needsUpdate = true;
	return table;
}

type RendererState = ReturnType<typeof THREE.RendererUtils.resetRendererState>;
/** A node builder's shared context, which three's declarations leave out. */
type SharedContext = Record<string, unknown>;

const drawingBufferSize = new THREE.Vector2();
// Undefined until the first draw, as three's own effects pass it.
let rendererState = undefined as unknown as RendererState;

const NORMAL_BITS = 5;
const NORMAL_MASK = (1 << NORMAL_BITS) - 1;
const NORMAL_ZERO = 1 << (NORMAL_BITS - 1);
const NORMAL_SCALE = NORMAL_ZERO - 1;

/**
 * The reference's packed value: linear depth as a float whose lowest ten
 * mantissa bits hold the normal instead, as two sphere-map coordinates of five
 * bits each. The value still reads as the depth, a hair off.
 */
function packDepthAndNormal(linearDepth: Node<'float'>, normal: Node<'vec3'>, dither: Node<'vec2'>): Node<'float'> {
	const coordinates = normal.xy
		.mul(TSL.inverseSqrt(normal.z.mul(2).add(2).max(0.0001)))
		.mul(NORMAL_SCALE)
		.add(dither)
		.floor()
		.add(NORMAL_ZERO)
		.clamp(0, NORMAL_MASK)
		.toConst();
	const bits = (TSL.floatBitsToUint(linearDepth) as unknown as Node<'uint'>)
		.bitAnd(TSL.uint(0xfffffc00))
		.bitOr(TSL.uint(coordinates.x))
		.bitOr(TSL.uint(coordinates.y).shiftLeft(TSL.uint(NORMAL_BITS)));
	return TSL.uintBitsToFloat(bits) as unknown as Node<'float'>;
}

/** A packed value's sphere-map coordinates, in steps of 1/15. */
function packedNormalCoordinates(packed: Node<'float'>): Node<'vec2'> {
	const bits = TSL.floatBitsToUint(packed) as unknown as Node<'uint'>;
	return TSL.vec2(
		TSL.float(bits.bitAnd(TSL.uint(NORMAL_MASK))),
		TSL.float(bits.shiftRight(TSL.uint(NORMAL_BITS)).bitAnd(TSL.uint(NORMAL_MASK))),
	).sub(NORMAL_ZERO);
}

function packedNormal(packed: Node<'float'>): Node<'vec3'> {
	const coordinates = packedNormalCoordinates(packed).div(NORMAL_SCALE).toConst();
	const z = TSL.float(1).sub(TSL.dot(coordinates, coordinates)).max(0).toConst();
	return TSL.vec3(coordinates.mul(z.sqrt().mul(2)), z.mul(2).sub(1));
}

/** Frisvad's orthonormal basis around a unit vector. */
function basis(normal: Node<'vec3'>): [Node<'vec3'>, Node<'vec3'>] {
	const a = TSL.float(1).div(normal.z.add(1)).toConst();
	const b = normal.x.mul(normal.y).mul(a).toConst();
	return [TSL.vec3(normal.x.mul(normal.x).mul(a).sub(1), b, normal.x), TSL.vec3(b, normal.y.mul(normal.y).mul(a).sub(1), normal.y)];
}

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
export class CortaoNode extends THREE.TempNode<'float'> {
	public static get type(): string {
		return 'CortaoNode';
	}

	/** While false nothing is drawn and the texture keeps its last contents. */
	public enabled = true;
	/** Depth samples per pixel; a change rebuilds the occlusion shader. */
	public sampleCount = 16;
	/** Frames seen since creation, whether or not occlusion was drawn. */
	public frames = 0;
	/** The occlusion radius, in world units. */
	public readonly radius = TSL.uniform(1);
	/** The occlusion's size as a fraction of the depth texture's; 0.5 is a quarter of the pixels. */
	public resolutionScale = 0.5;
	/**
	 * How much of each new frame goes into the history, 1 for none kept. Lower is
	 * smoother and slower to follow moving objects; history the camera moved away
	 * from, by depth, is dropped.
	 */
	public readonly temporalBlend = TSL.uniform(0.1);

	private readonly packedTarget = new THREE.RenderTarget(1, 1, {
		depthBuffer: false,
		// One 32-bit float per pixel: sampling cost follows the bytes read.
		format: THREE.RedFormat,
		type: THREE.FloatType,
		minFilter: THREE.NearestFilter,
		magFilter: THREE.NearestFilter,
	});
	private readonly occlusionTarget = new THREE.RenderTarget(1, 1, {
		depthBuffer: false,
		minFilter: THREE.NearestFilter,
		magFilter: THREE.NearestFilter,
	});
	private readonly filteredTarget = new THREE.RenderTarget(1, 1, {
		depthBuffer: false,
		minFilter: THREE.NearestFilter,
		magFilter: THREE.NearestFilter,
	});
	/**
	 * Accumulated occlusion in red and its linear depth in green, in turns: each
	 * frame writes one from the other, and the texture node reads the one written.
	 */
	private readonly resolvedTargets = [0, 1].map(
		() =>
			new THREE.RenderTarget(1, 1, {
				depthBuffer: false,
				format: THREE.RGFormat,
				// Half floats filter everywhere; 32-bit ones need an optional WebGPU feature.
				type: THREE.HalfFloatType,
			}),
	);
	private readonly lookupTable = createOccluderLookupTexture();
	private readonly packMaterial = new THREE.NodeMaterial();
	private readonly occlusionMaterial = new THREE.NodeMaterial();
	private readonly filterMaterial = new THREE.NodeMaterial();
	/** One per resolved target it writes, reading the other. */
	private readonly temporalMaterials = [new THREE.NodeMaterial(), new THREE.NodeMaterial()];
	private readonly textureNode: TextureNode;
	/** One quad per pass: a quad whose material changes every draw is set up again every draw. */
	private readonly quads = new Map<THREE.NodeMaterial, THREE.QuadMesh>();
	/** Which resolved target this frame writes. */
	private parity = 0;

	private readonly projection = TSL.uniform(new THREE.Matrix4());
	private readonly projectionInverse = TSL.uniform(new THREE.Matrix4());
	private readonly near = TSL.uniform(0.1);
	private readonly far = TSL.uniform(1000);
	/** Texture coordinates and linear depth to view space: xy scale, zw offset. */
	private readonly unprojectParams = TSL.uniform(new THREE.Vector4());
	private readonly normalBias = TSL.uniform(0);
	private readonly maxApparentCircleRadiusCoefficient = TSL.uniform(0);
	/** Shifts the per-pixel noise every frame, so the history averages different samples. */
	private readonly noiseOffset = TSL.uniform(new THREE.Vector2());
	private readonly cameraWorld = TSL.uniform(new THREE.Matrix4());
	private readonly previousViewProjection = TSL.uniform(new THREE.Matrix4());
	/** 0 when the history is from another size or there is none yet. */
	private readonly historyValid = TSL.uniform(0);
	private readonly targetSize = new THREE.Vector2();

	private builtSampleCount = -1;
	private buildOcclusion: (() => Node) | null = null;
	private sharedContext: SharedContext = {};

	/**
	 * @param depthNode A scene pass's depth texture.
	 * @param camera Read every frame, so a pass that changes cameras is followed.
	 */
	public constructor(
		private readonly depthNode: TextureNode,
		private readonly camera: () => THREE.Camera,
	) {
		super('float');
		this.updateBeforeType = THREE.NodeUpdateType.FRAME;
		this.packMaterial.name = 'CORTAO.pack';
		this.occlusionMaterial.name = 'CORTAO.occlusion';
		this.filterMaterial.name = 'CORTAO.filter';
		for (const material of this.temporalMaterials) material.name = 'CORTAO.temporal';
		for (const material of [this.packMaterial, this.occlusionMaterial, this.filterMaterial, ...this.temporalMaterials]) {
			const quad = new THREE.QuadMesh(material);
			quad.name = material.name;
			this.quads.set(material, quad);
		}
		this.textureNode = TSL.passTexture(this as unknown as THREE.PassNode, this.resolvedTargets[0].texture);
	}

	/** The filtered, accumulated occlusion in red. */
	public getTextureNode(): TextureNode {
		return this.textureNode;
	}

	private viewPosition(uv: Node<'vec2'>, linearDepth: Node<'float'>): Node<'vec3'> {
		const ndc = TSL.vec2(uv.x, uv.y.oneMinus()).mul(2).sub(1);
		return TSL.vec3(ndc.mul(this.unprojectParams.xy).add(this.unprojectParams.zw), -1).mul(linearDepth);
	}

	/** The view-space normal rebuilt from depth; the scene pass has no normal buffer. */
	private normalFromDepth(uv: Node<'vec2'>): Node<'vec3'> {
		return TSL.getNormalFromDepth(
			uv,
			// It takes the texture itself; its declaration names a node.
			this.depthNode.value as unknown as Node,
			this.projectionInverse,
		);
	}

	/** Linear depth and normal packed into the red channel; 0 on the background. */
	private pack(): Node {
		const uv = TSL.uv();
		const depth = TSL.projectionDepth(this.depthNode.sample(uv).r, this.near, this.far).toConst();
		const position = TSL.getViewPosition(uv, depth, this.projectionInverse);
		const dither = TSL.vec2(TSL.rand(uv.add(this.noiseOffset)), TSL.rand(uv.add(this.noiseOffset).add(TSL.vec2(0.4321, 0.8765))));
		const packed = packDepthAndNormal(position.z.negate(), this.normalFromDepth(uv), dither);
		return TSL.vec4(TSL.isBackgroundDepth(depth).select(0, packed), 0, 0, 1);
	}

	/** The reference's main pass, with the sample loop unrolled. */
	private occlusion(sampleCount: number): Node {
		const packed = TSL.texture(this.packedTarget.texture);
		const uv = TSL.uv();
		const size = TSL.vec2(TSL.textureSize(packed) as unknown as Node<'ivec2'>).toConst();
		const load = (coordinates: Node<'vec2'>) => packed.load(TSL.ivec2(coordinates.clamp(0, 0.99999).mul(size)));

		const center = load(uv).r.toConst();
		const linearDepth = center;
		// Full precision for the pixel itself; only its neighbors use the packed normal.
		const normal = this.normalFromDepth(uv).toConst();

		const noiseUv = uv.add(this.noiseOffset).toConst();
		const rotation = TSL.interleavedGradientNoise(TSL.screenCoordinate.xy.add(this.noiseOffset.mul(5.588238)))
			.mul(Math.PI * 2)
			.toConst();
		const offsetDither = TSL.rand(noiseUv).toConst();
		const tangentRotation = TSL.rand(noiseUv.add(TSL.vec2(0.1234, 0.5678)))
			.mul(Math.PI * 2)
			.toConst();
		const flip = TSL.rand(noiseUv.add(TSL.vec2(0.77, 0.31))).lessThan(0.5);

		// The world radius, limited to a fraction of the screen height.
		const clampedRadius = TSL.min(this.radius, linearDepth.mul(this.maxApparentCircleRadiusCoefficient)).toConst();
		// Near the sampling sphere it covers more of the screen; widen the sampled circle to cover it.
		const apparentCircleRadius = clampedRadius.mul(TSL.inverseSqrt(linearDepth.mul(linearDepth).sub(clampedRadius.mul(clampedRadius))));
		const circleRadius = apparentCircleRadius.mul(linearDepth).toConst();

		const unbiased = this.viewPosition(uv, linearDepth);
		const viewDirection = TSL.normalize(unbiased.negate()).toConst();
		// Pushed out along the normal to avoid self-shadowing.
		const position = unbiased.add(normal.mul(this.normalBias.mul(linearDepth))).toConst();

		const [rightBase, upBase] = basis(viewDirection);
		// A random clockwise or counter-clockwise spiral.
		const right = flip.select(rightBase.negate(), rightBase).toConst();
		const up = upBase.toConst();
		const scaledSin = TSL.sin(rotation).mul(circleRadius).toConst();
		const scaledCos = TSL.cos(rotation).mul(circleRadius).toConst();

		const [tangentBase, bitangentBase] = basis(normal);
		const tangentSin = TSL.sin(tangentRotation).toConst();
		const tangentCos = TSL.cos(tangentRotation).toConst();
		const tangent = tangentBase.mul(tangentCos).add(bitangentBase.mul(tangentSin)).toConst();
		const bitangent = bitangentBase.mul(tangentCos).sub(tangentBase.mul(tangentSin)).toConst();

		const rayBits = TSL.uint(0).toVar();
		// How strongly samples gather at the center.
		const skew = 3.5;
		for (let i = 0; i < sampleCount; i++) {
			const angle = i * GOLDEN_ANGLE;
			const innerRadius = i === 0 ? 0 : (i / sampleCount) ** (skew * 0.5);
			const outerRadius = ((i + 1) / sampleCount) ** (skew * 0.5);
			const sampleWeight = Math.sqrt(outerRadius * outerRadius - innerRadius * innerRadius);
			const spiral = new THREE.Vector2(Math.cos(angle), Math.sin(angle));
			const inner = spiral.clone().multiplyScalar(innerRadius);
			const span = spiral.clone().multiplyScalar(outerRadius - innerRadius);

			const offset = TSL.vec2(span).mul(offsetDither).add(TSL.vec2(inner));
			const circle = TSL.vec2(scaledSin.mul(offset.x).add(scaledCos.mul(offset.y)), scaledSin.mul(offset.y).sub(scaledCos.mul(offset.x)));
			const samplePosition = right.mul(circle.x).add(up.mul(circle.y)).add(position);
			const sampleUv = TSL.getScreenPositionFromClip(this.projection.mul(TSL.vec4(samplePosition, 1))).toConst();
			const onScreen = sampleUv.x
				.greaterThanEqual(0)
				.and(sampleUv.x.lessThan(1))
				.and(sampleUv.y.greaterThanEqual(0))
				.and(sampleUv.y.lessThan(1));
			const sample = load(sampleUv).r.toConst();
			// Background and off-screen samples have depth 0, which is never inside the hemisphere.
			const sampleDepth = onScreen.select(sample, 0);

			const delta = this.viewPosition(sampleUv, sampleDepth).sub(position).toConst();
			const lengthSquared = TSL.dot(delta, delta).toConst();
			const inside = lengthSquared.lessThan(clampedRadius.mul(clampedRadius)).and(TSL.dot(delta, normal).greaterThan(0));
			// Samples outside the hemisphere skip the rest, as in the reference.
			TSL.If(inside, () => {
				const inverseLength = TSL.inverseSqrt(lengthSquared.max(1e-12)).toConst();
				const direction = delta.mul(inverseLength).toConst();
				const sampleNormal = packedNormal(sample).toConst();

				const slopeCompensation = TSL.inverseSqrt(TSL.dot(sampleNormal, viewDirection).max(1 / DEFAULTS.maxSlopeCompensation)).toConst();
				const occluderRadius = clampedRadius.mul(sampleWeight).mul(slopeCompensation).mul(inverseLength).toConst();

				// A front-facing occluder moves one radius into its surface so it does not shadow itself;
				// a back-facing one grows instead, so light does not leak around it.
				const backface = TSL.dot(sampleNormal, direction).greaterThanEqual(DEFAULTS.backfaceThreshold);
				const inward = occluderRadius.negate().div(TSL.float(1).sub(occluderRadius.mul(occluderRadius).mul(0.5)).max(0.000001));
				const occluderDirection = backface.select(direction, TSL.normalize(sampleNormal.mul(inward).add(direction))).toConst();
				const radius = backface.select(occluderRadius.mul(TSL.mix(1, slopeCompensation, DEFAULTS.backfaceBoost)), occluderRadius);

				const hemisphere = TSL.vec3(
					TSL.dot(occluderDirection, tangent),
					TSL.dot(occluderDirection, bitangent),
					TSL.dot(occluderDirection, normal),
				).toConst();
				const sphereMap = hemisphere.xy.mul(TSL.inverseSqrt(hemisphere.z.mul(8).add(8).max(0.0001))).add(0.5);
				const entry = TSL.texture3D(this.lookupTable, TSL.vec3(sphereMap, radius.mul(Math.SQRT1_2)), TSL.int(0)).toConst();
				const byte = (channel: Node<'float'>, shift: number) => TSL.uint(channel.mul(255).add(0.5)).shiftLeft(TSL.uint(shift));
				const blocked = byte(entry.r, 0).bitOr(byte(entry.g, 8)).bitOr(byte(entry.b, 16)).bitOr(byte(entry.a, 24));
				// The sphere map has no entry for the direction straight down.
				rayBits.assign(rayBits.bitOr(hemisphere.z.greaterThan(-0.99).select(blocked, TSL.uint(0))));
			});
		}

		const occlusion = TSL.float(1).sub(TSL.float(TSL.countOneBits(rayBits) as unknown as Node<'uint'>).div(RAY_COUNT));
		return TSL.vec4(linearDepth.lessThanEqual(0).select(1, occlusion));
	}

	/** The reference's sparse filter: neighbors weighted by how well they fit the pixel's depth plane and normal. */
	private filter(): Node {
		const packed = TSL.texture(this.packedTarget.texture);
		const occlusion = TSL.texture(this.occlusionTarget.texture);
		const size = TSL.ivec2(TSL.textureSize(packed) as unknown as Node<'ivec2'>).toConst();
		const pixel = TSL.ivec2(TSL.uv().mul(TSL.vec2(size))).toConst();
		// Integer vectors clamp in both shading languages; the declarations only know floats.
		const clamped = (coordinates: Node<'ivec2'>) =>
			(coordinates as unknown as Node<'vec2'>).clamp(
				TSL.ivec2(0, 0) as unknown as Node<'vec2'>,
				size.sub(1) as unknown as Node<'vec2'>,
			) as unknown as Node<'ivec2'>;
		const onScreen = (coordinates: Node<'ivec2'>) =>
			coordinates.x
				.greaterThanEqual(0)
				.and(coordinates.y.greaterThanEqual(0))
				.and(coordinates.x.lessThan(size.x))
				.and(coordinates.y.lessThan(size.y));
		const depthAt = (coordinates: Node<'ivec2'>) => onScreen(coordinates).select(packed.load(clamped(coordinates)).r, 0);

		const center = packed.load(pixel).r.toConst();
		const inverseDepth = TSL.float(1).div(center).toConst();
		const centerNormal = packedNormalCoordinates(center).toConst();
		const sensitivity = DEFAULTS.filterDepthSensitivity;

		// Depth slopes from the pixel's 2x2 quad: even pixel minus odd pixel.
		const quadOrigin = pixel.div(2).mul(2).toConst();
		const deltaX = depthAt(TSL.ivec2(quadOrigin.x, pixel.y))
			.sub(depthAt(TSL.ivec2(quadOrigin.x.add(1), pixel.y)))
			.mul(inverseDepth)
			.mul(sensitivity)
			.toConst();
		const deltaY = depthAt(TSL.ivec2(pixel.x, quadOrigin.y))
			.sub(depthAt(TSL.ivec2(pixel.x, quadOrigin.y.add(1))))
			.mul(inverseDepth)
			.mul(sensitivity)
			.toConst();

		const weightAt = (x: number, y: number) => {
			const coordinates = pixel.add(TSL.ivec2(x, y));
			const sample = packed.load(clamped(coordinates)).r.toConst();
			const depth = onScreen(coordinates).select(sample, 0);
			const depthError = deltaX.mul(x).add(deltaY.mul(y)).add(depth.mul(inverseDepth).mul(sensitivity).sub(sensitivity)).abs();
			const normalDelta = packedNormalCoordinates(sample).sub(centerNormal);
			const normalError = TSL.dot(normalDelta, normalDelta).mul(DEFAULTS.filterNormalSensitivity / 32);
			return TSL.exp2(depthError.add(normalError).negate());
		};
		const occlusionAt = (x: number, y: number) => occlusion.load(clamped(pixel.add(TSL.ivec2(x, y)))).r;

		const sum = occlusion.load(pixel).r.toVar();
		const weightSum = TSL.float(1).toVar();
		for (const [x, y] of FILTER_OFFSETS) {
			// The smaller of a mirrored pair's weights goes to both, so one bad side cannot pull the result.
			const weight = TSL.min(weightAt(-x, -y), weightAt(x, y)).toConst();
			weightSum.addAssign(weight.mul(2));
			sum.addAssign(occlusionAt(-x, -y).add(occlusionAt(x, y)).mul(weight));
		}
		return TSL.vec4(center.lessThanEqual(0).select(1, sum.div(weightSum)));
	}

	/**
	 * The filtered occlusion blended into last frame's, found where this pixel's
	 * surface was then. The history is dropped where its depth does not match
	 * that place, off screen, and on the background.
	 */
	private temporal(history: TextureNode): Node {
		const packed = TSL.texture(this.packedTarget.texture);
		const filtered = TSL.texture(this.filteredTarget.texture);
		const uv = TSL.uv();
		const size = TSL.vec2(TSL.textureSize(packed) as unknown as Node<'ivec2'>).toConst();
		const pixel = TSL.ivec2(uv.mul(size)).toConst();
		const linearDepth = packed.load(pixel).r.toConst();
		const current = filtered.load(pixel).r.toConst();

		const world = this.cameraWorld.mul(TSL.vec4(this.viewPosition(uv, linearDepth), 1));
		const clip = this.previousViewProjection.mul(world).toConst();
		const previousUv = TSL.getScreenPositionFromClip(clip).toConst();
		const onScreen = previousUv.x
			.greaterThanEqual(0)
			.and(previousUv.x.lessThan(1))
			.and(previousUv.y.greaterThanEqual(0))
			.and(previousUv.y.lessThan(1));
		const previous = history.load(TSL.ivec2(previousUv.clamp(0, 0.99999).mul(size))).toConst();
		// For a perspective camera, clip w is the linear depth that place had.
		const depthMatches = previous.g.sub(clip.w).abs().lessThan(clip.w.mul(0.05));
		const keep = onScreen.and(depthMatches).and(this.historyValid.greaterThan(0.5)).and(previous.g.greaterThan(0));
		const blend = keep.select(this.temporalBlend, 1);
		const occlusion = TSL.mix(previous.r, current, blend);
		return TSL.vec4(linearDepth.lessThanEqual(0).select(1, occlusion), linearDepth, 0, 1);
	}

	private setSize(width: number, height: number): boolean {
		if (this.targetSize.x === width && this.targetSize.y === height) return false;
		this.targetSize.set(width, height);
		for (const target of [this.packedTarget, this.occlusionTarget, this.filteredTarget, ...this.resolvedTargets])
			target.setSize(width, height);
		return true;
	}

	public override updateBefore(frame: NodeFrame): undefined {
		this.frames++;
		const renderer = frame.renderer;
		const camera = this.camera();
		if (!this.enabled || !renderer || !this.buildOcclusion || !(camera instanceof THREE.PerspectiveCamera)) return undefined;

		// The scene's own size, which may be below the drawing buffer's (dynamic resolution).
		const depth = this.depthNode.value.image as { width?: number; height?: number } | undefined;
		const full =
			depth?.width && depth.height ? drawingBufferSize.set(depth.width, depth.height) : renderer.getDrawingBufferSize(drawingBufferSize);
		const scale = THREE.MathUtils.clamp(this.resolutionScale, 0.25, 1);
		const size = { width: Math.max(1, Math.round(full.x * scale)), height: Math.max(1, Math.round(full.y * scale)) };
		if (this.setSize(size.width, size.height)) this.historyValid.value = 0;

		// The reference's populateMainPassParameters.
		const inverse = camera.projectionMatrixInverse.elements;
		this.projection.value.copy(camera.projectionMatrix);
		this.projectionInverse.value.copy(camera.projectionMatrixInverse);
		this.near.value = camera.near;
		this.far.value = camera.far;
		this.unprojectParams.value.set(inverse[0], inverse[5], inverse[12], inverse[13]);
		this.normalBias.value = (inverse[0] / size.width) * Math.SQRT2;
		const maxApparentCircleRadius = inverse[5] * DEFAULTS.maxScreenspaceRadiusHeightRatio * 2;
		this.maxApparentCircleRadiusCoefficient.value = maxApparentCircleRadius / Math.hypot(maxApparentCircleRadius, 1);

		// Each frame samples other directions; the history averages them.
		const step = this.frames % 64;
		this.noiseOffset.value.set((step * 0.618034) % 1, (step * 0.754878) % 1);
		camera.updateMatrixWorld();
		this.cameraWorld.value.copy(camera.matrixWorld);

		if (this.sampleCount !== this.builtSampleCount) {
			this.builtSampleCount = this.sampleCount;
			this.occlusionMaterial.fragmentNode = this.buildOcclusion();
			this.occlusionMaterial.needsUpdate = true;
		}

		rendererState = THREE.RendererUtils.resetRendererState(renderer, rendererState);
		renderer.setClearColor(0xffffff, 1);
		this.parity = 1 - this.parity;
		const resolved = this.resolvedTargets[this.parity];
		for (const [material, target] of [
			[this.packMaterial, this.packedTarget],
			[this.occlusionMaterial, this.occlusionTarget],
			[this.filterMaterial, this.filteredTarget],
			[this.temporalMaterials[this.parity], resolved],
		] as const) {
			renderer.setRenderTarget(target);
			this.quads.get(material)?.render(renderer);
		}
		THREE.RendererUtils.restoreRendererState(renderer, rendererState);
		this.textureNode.value = resolved.texture;
		this.previousViewProjection.value.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
		this.historyValid.value = 1;
		return undefined;
	}

	public override setup(builder: NodeBuilder): TextureNode {
		this.sharedContext = (builder as unknown as { getSharedContext(): SharedContext }).getSharedContext();
		const shared = () => TSL.context(this.sharedContext);
		this.buildOcclusion = TSL.Fn(() => this.occlusion(this.sampleCount));
		this.builtSampleCount = this.sampleCount;
		this.packMaterial.contextNode = shared();
		this.packMaterial.fragmentNode = TSL.Fn(() => this.pack())();
		this.packMaterial.needsUpdate = true;
		this.occlusionMaterial.contextNode = shared();
		this.occlusionMaterial.fragmentNode = this.buildOcclusion();
		this.occlusionMaterial.needsUpdate = true;
		this.filterMaterial.contextNode = shared();
		this.filterMaterial.fragmentNode = TSL.Fn(() => this.filter())();
		this.filterMaterial.needsUpdate = true;
		this.temporalMaterials.forEach((material, index) => {
			const history = TSL.texture(this.resolvedTargets[1 - index].texture);
			material.contextNode = shared();
			material.fragmentNode = TSL.Fn(() => this.temporal(history))();
			material.needsUpdate = true;
		});
		return this.textureNode;
	}

	public override dispose(): void {
		super.dispose();
		this.packedTarget.dispose();
		this.occlusionTarget.dispose();
		this.filteredTarget.dispose();
		for (const target of this.resolvedTargets) target.dispose();
		this.lookupTable.dispose();
		this.packMaterial.dispose();
		this.occlusionMaterial.dispose();
		this.filterMaterial.dispose();
		for (const material of this.temporalMaterials) material.dispose();
	}
}
