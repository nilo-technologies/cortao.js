// CORTAO for three.js's WebGLRenderer (WebGL2).
//
// A fragment-shader port of the CORTAO reference implementation v1.0.1 by
// Daniel Isheden, https://github.com/theagentd/CORTAO (src/shaders/*.h and
// src/CORTAOParameters.h). The reference is three compute passes; WebGL2 has no
// compute, so each pass is a full-screen fragment pass here. Differences from
// the reference are marked "PORT:" below.
//
// MIT License
//
// Copyright (c) 2026 Daniel Isheden
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { createOccluderLookupTable } from '../lookupTable.js';
export const VERTEX = /* glsl */ `
in vec3 position;
void main() {
	gl_Position = vec4(position.xy, 0.0, 1.0);
}`;
// CORTAOBase.h
const BASE = /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
precision highp usampler3D;

#define CORTAO_PACKED_NORMAL_BITS 10

const float CORTAO_PI2 = 6.283185307179586476925286766559;
const float CORTAO_GOLDEN_ANGLE = 2.3999632297286533222315555066336;

uint cortao_seed(uvec2 coords) {
	uvec2 q = coords * uvec2(1719413u * 929u, 140473u * 2467u * 11u);
	return q.x ^ q.y;
}

uint cortao_hash(uint n) {
	return n * (n ^ (n >> 15u));
}

uvec2 cortao_hash(uvec2 n) {
	return n * (n ^ (n >> 15u));
}

uint cortao_encodeNormal(vec3 normal, vec2 dither) {
	uint bitsPerComponent = uint(CORTAO_PACKED_NORMAL_BITS / 2);
	uint componentMask = (1u << bitsPerComponent) - 1u;
	uint componentMax = componentMask >> 1u;

	vec2 encodedNormal = normal.xy * inversesqrt(normal.z * 2.0 + 2.0);
	ivec2 values = ivec2(floor(encodedNormal * float(componentMax) + dither));

	return (uint(values.x) & componentMask) | ((uint(values.y) & componentMask) << bitsPerComponent);
}

ivec2 cortao_unpackNormalXY(uint packedData) {
	int bitsPerComponent = CORTAO_PACKED_NORMAL_BITS / 2;
	ivec2 leftShift = ivec2(32 - bitsPerComponent, 32 - bitsPerComponent * 2);
	return (ivec2(int(packedData)) << leftShift) >> (32 - bitsPerComponent);
}

vec3 cortao_decodeNormal(uint packedData) {
	uint componentMax = ((1u << uint(CORTAO_PACKED_NORMAL_BITS / 2)) - 1u) >> 1u;
	vec2 e = vec2(cortao_unpackNormalXY(packedData)) * (1.0 / float(componentMax));
	// PORT: max() guards sqrt against a slightly negative z from quantization.
	float z = max(1.0 - e.x * e.x - e.y * e.y, 0.0);
	return vec3(e * (2.0 * sqrt(z)), z * 2.0 - 1.0);
}

float cortao_unpackDepth(uint packedData) {
	return uintBitsToFloat(packedData);
}

// PORT: GLSL ES 3.00 has no bitCount().
uint cortao_bitCount(uint v) {
	v = v - ((v >> 1u) & 0x55555555u);
	v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
	return (((v + (v >> 4u)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24u;
}
`;
// CORTAOPrepass.h
// PORT: the reference writes each pixel into up to 8 mip levels, where every mip
// texel keeps one stochastically chosen child. That choice is a pure function of
// the texel coordinates and the seed, so the main pass walks it back down to mip
// 0 (see cortao_samplePackedData) and this pass only writes mip 0. Results are
// identical; only the cache benefit of real mips is lost.
const PREPASS_FRAGMENT = /* glsl */ `
${BASE}
uniform sampler2D tDepth;
uniform sampler2D tNormal;
uniform float depthScale;
uniform float depthOffset;
uniform uint randomSeed;

out highp uint outPacked;

void main() {
	ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
	float depth = texelFetch(tDepth, pixelCoords, 0).r;

	uint packedData = 0u;

	float skyBoxDepth = float(depthScale < 0.0);
	if (depth != skyBoxDepth) {
		float linearDepth = 1.0 / (depth * depthScale + depthOffset);

		// View-space normal, stored as rgb = n * 0.5 + 0.5 by MeshNormalMaterial.
		vec3 normal = normalize(texelFetch(tNormal, pixelCoords, 0).xyz * 2.0 - 1.0);

		uint seed = cortao_hash(cortao_seed(uvec2(pixelCoords)) + randomSeed);
		vec2 dither = vec2(seed & 0xFFFFu, seed >> 16u) * (1.0 / 65535.0);
		uint mask = (1u << CORTAO_PACKED_NORMAL_BITS) - 1u;
		uint packedNormal = cortao_encodeNormal(normal, dither);

		packedData = (floatBitsToUint(linearDepth) & ~mask) | (packedNormal & mask);
	}

	outPacked = packedData;
}`;
// CORTAOMainPass.h, perspective projection.
const MAIN_FRAGMENT = /* glsl */ `
${BASE}
uniform usampler2D tPacked;
uniform sampler2D tNormal;
#if CORTAO_OCCLUDER_LOOKUP_TABLE
uniform usampler3D tOccluderLookupTable;
#endif

uniform ivec2 resolution;
uniform int mipCount;
uniform uint randomSeed;
uniform float normalBias;
uniform float radius;
uniform vec4 unprojectParams;
uniform vec4 projectionParams;
uniform float maxApparentCircleRadiusCoefficient;
uniform float mipBias;
uniform float inverseMaxSlopeCompensation;
uniform float backfaceThreshold;
uniform float backfaceBoost;

out vec4 outColor;

// NEAREST min/mag/mip filtering with CLAMP_TO_BORDER (border = 0).
uint cortao_samplePackedData(vec2 texCoords, float mipmap) {
	if (texCoords.x < 0.0 || texCoords.y < 0.0 || texCoords.x >= 1.0 || texCoords.y >= 1.0) {
		return 0u;
	}
	int level = clamp(int(ceil(mipmap + 0.5)) - 1, 0, mipCount - 1);
	ivec2 levelResolution = max(resolution >> level, ivec2(1));
	uvec2 coords = uvec2(min(ivec2(texCoords * vec2(levelResolution)), levelResolution - 1));

	// PORT: walk the prepass's stochastic mip chain down to the mip 0 pixel that
	// the reference would have copied into this mip texel.
	for (int l = level; l >= 1; l--) {
		uvec2 dither = uvec2(randomSeed) >> uvec2(l * 2, l * 2 + 1);
		coords = coords * 2u + ((coords.yx ^ dither) & 1u);
	}
	return texelFetch(tPacked, ivec2(coords), 0).r;
}

vec3 cortao_loadNormal(ivec2 pixelCoords) {
	return normalize(texelFetch(tNormal, pixelCoords, 0).xyz * 2.0 - 1.0);
}

vec3 cortao_rayDirection(uint rayIndex) {
	float angle = float(rayIndex) * CORTAO_GOLDEN_ANGLE;
	float r = sqrt((float(rayIndex) + 0.5) / float(CORTAO_RAY_COUNT));
	return vec3(cos(angle) * r, sin(angle) * r, sqrt(1.0 - r * r));
}

vec3 cortao_unprojectWithOffset(vec2 texCoords, float linearDepth, vec3 offset) {
	vec2 direction = texCoords * unprojectParams.xy + unprojectParams.zw;
	return vec3(direction, -1.0) * linearDepth + offset;
}

uint cortao_computeRayOcclusion(
	float offsetDither,
	mat3 offsetTransform,
	mat3 tangentSpaceTransform,
	float baseMipBias,
	vec3 viewPosition,
	vec3 normal,
	vec3 viewDirection,
	float maxRadius
) {
	uint rayBits = 0u;

	for (int i = 0; i < CORTAO_SAMPLE_COUNT; i++) {
		// Golden angle spiral direction.
		float a = float(i) * CORTAO_GOLDEN_ANGLE;
		vec2 spiral = vec2(cos(a), sin(a));

		float inverseSampleCount = 1.0 / float(CORTAO_SAMPLE_COUNT);

		// How much the sample distribution is pulled into the center.
		float skew = 3.5;

		float innerRadius = i == 0 ? 0.0 : pow(float(i) * inverseSampleCount, skew * 0.5);
		float outerRadius = pow((float(i) + 1.0) * inverseSampleCount, skew * 0.5);

		vec2 innerSpiral = spiral * innerRadius;
		vec2 outerSpiral = spiral * outerRadius;

		float sampleArea = outerRadius * outerRadius - innerRadius * innerRadius;
		float sampleWeight = sqrt(sampleArea);

		float sampleMipmap = log2(outerRadius);

		vec2 sampleOffset = (outerSpiral - innerSpiral) * offsetDither + innerSpiral;

		vec3 projectedCoords = offsetTransform * vec3(sampleOffset, 1.0);
		vec2 sampleCoords = projectedCoords.xy / projectedCoords.z;

		float mipmap = baseMipBias + sampleMipmap;

		uint packedData = cortao_samplePackedData(sampleCoords, mipmap);
		float sampleDepth = cortao_unpackDepth(packedData);

		// The sample's view-space position relative to the center pixel.
		vec3 dPos = cortao_unprojectWithOffset(sampleCoords, sampleDepth, -viewPosition);

		// Skybox and off-screen samples have a linear depth of 0, which is always
		// outside the hemisphere.
		float lengthSqrd = dot(dPos, dPos);
		if (lengthSqrd < maxRadius * maxRadius && dot(dPos, normal) > 0.0) {
			float inverseLength = inversesqrt(lengthSqrd);
			vec3 direction = dPos * inverseLength;

			vec3 sampleNormal = cortao_decodeNormal(packedData);

			float slopeCompensation = inversesqrt(max(dot(sampleNormal, viewDirection), inverseMaxSlopeCompensation));
			float occluderRadius = (maxRadius * sampleWeight) * (slopeCompensation * inverseLength);

			float sampleDot = dot(sampleNormal, direction);
			if (sampleDot < backfaceThreshold) {
				// Not a backface occluder: move the circle one radius into the
				// occluder to prevent self-occlusion.
				float adjustedOffset = -occluderRadius / max(1.0 - occluderRadius * occluderRadius * 0.5, 0.000001);
				direction = normalize(sampleNormal * adjustedOffset + direction);
			} else {
				occluderRadius *= mix(1.0, slopeCompensation, backfaceBoost);
			}

			vec3 hemisphereDirection = direction * tangentSpaceTransform;

			uint occluderRayBits;
			#if CORTAO_OCCLUDER_LOOKUP_TABLE
				float mul = inversesqrt(hemisphereDirection.z * 8.0 + 8.0);
				vec2 sphereMapCoordinates = hemisphereDirection.xy * mul + 0.5;
				vec3 lookupCoords = vec3(sphereMapCoordinates, occluderRadius * sqrt(0.5));
				occluderRayBits = textureLod(tOccluderLookupTable, lookupCoords, 0.0).r
					& (hemisphereDirection.z > -0.99 ? 0xFFFFFFFFu : 0u);
			#else
				float occluderCosAngle = max(0.0, 1.0 - occluderRadius * occluderRadius * 0.5);
				occluderRayBits = 0u;
				for (uint j = 0u; j < uint(CORTAO_RAY_COUNT); j++) {
					if (dot(cortao_rayDirection(j), hemisphereDirection) > occluderCosAngle) {
						occluderRayBits |= 1u << j;
					}
				}
			#endif

			rayBits |= occluderRayBits;
		}
	}

	return rayBits;
}

// Frisvad's orthonormal basis.
void cortao_constructBasis(vec3 normal, out vec3 tangent, out vec3 bitangent) {
	float a = 1.0 / (1.0 + normal.z);
	float b = normal.x * normal.y * a;
	tangent = vec3(normal.x * normal.x * a - 1.0, b, normal.x);
	bitangent = vec3(b, normal.y * normal.y * a - 1.0, normal.y);
}

void main() {
	ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
	vec2 texCoords = (vec2(pixelCoords) + 0.5) / vec2(resolution);

	float linearDepth = cortao_unpackDepth(texelFetch(tPacked, pixelCoords, 0).r);
	if (linearDepth == 0.0) {
		outColor = vec4(1.0);
		return;
	}

	vec3 normal = cortao_loadNormal(pixelCoords);
	vec3 viewPosition = cortao_unprojectWithOffset(texCoords, linearDepth, vec3(0.0));

	uvec2 random = cortao_hash(cortao_seed(uvec2(pixelCoords)) + uvec2(randomSeed & 0xFFFFu, randomSeed >> 16u));

	float randomSampleRotation = float(random.x) * (CORTAO_PI2 / float(0xFFFFFFFFu));
	float offsetDither = float(random.y) * (1.0 / float(0xFFFFFFFFu));
	float randomTangentRotation = float(random.x ^ random.y ^ randomSeed) * (CORTAO_PI2 / float(0xFFFFFFFFu));
	vec2 sampleRotation = vec2(sin(randomSampleRotation), cos(randomSampleRotation));

	// Enforce the maximum screen-space radius as a linear function of depth.
	float clampedRadius = min(radius, linearDepth * maxApparentCircleRadiusCoefficient);

	// Close to the sampling sphere it covers more of the screen; expand the
	// sampled circle so it covers the whole sphere.
	float inverseCircleDistance = inversesqrt(linearDepth * linearDepth - clampedRadius * clampedRadius);
	float apparentCircleRadius = clampedRadius * inverseCircleDistance;
	float circleRadius = apparentCircleRadius * linearDepth;

	vec3 viewDirection = -normalize(viewPosition);

	float stretchMultiplier = inversesqrt(viewDirection.z);
	float baseMipBias = log2(apparentCircleRadius * stretchMultiplier) + mipBias;

	// Push the center out along the normal to avoid self-shadowing.
	float pixelSize = normalBias * linearDepth;
	viewPosition = normal * pixelSize + viewPosition;

	vec2 scaledRotation = sampleRotation * circleRadius;
	mat3 rotationScaleMatrix = mat3(
		scaledRotation.x, -scaledRotation.y, 0.0,
		scaledRotation.y, scaledRotation.x, 0.0,
		0.0, 0.0, 1.0
	);

	vec3 right;
	vec3 up;
	cortao_constructBasis(viewDirection, right, up);

	// Randomly flip between a clockwise and counter-clockwise spiral.
	if ((random.x & 1u) == 0u) {
		right = -right;
	}

	mat3 circleMatrix = mat3(right, up, viewPosition);

	mat3 projectionMatrix = mat3(
		projectionParams.x, 0.0, 0.0,
		0.0, projectionParams.y, 0.0,
		projectionParams.zw, -1.0
	);

	mat3 offsetTransform = projectionMatrix * (circleMatrix * rotationScaleMatrix);

	vec3 t;
	vec3 bt;
	cortao_constructBasis(normal, t, bt);
	float s = sin(randomTangentRotation);
	float c = cos(randomTangentRotation);
	vec3 tangent = c * t + s * bt;
	vec3 bitangent = c * bt - s * t;
	mat3 tangentSpaceTransform = mat3(tangent, bitangent, normal);

	uint rayBits = cortao_computeRayOcclusion(
		offsetDither, offsetTransform, tangentSpaceTransform, baseMipBias,
		viewPosition, normal, viewDirection, clampedRadius
	);

	// 1.0 - occluded rays / total rays
	float occlusion = 1.0 - float(cortao_bitCount(rayBits)) / float(CORTAO_RAY_COUNT);

	// Bent normal: the average unoccluded ray, back in view space.
	vec3 bentNormal = vec3(0.0);
	for (uint i = 0u; i < uint(CORTAO_RAY_COUNT); i++) {
		bentNormal += cortao_rayDirection(i) * ((rayBits & (1u << i)) == 0u ? 1.0 : 0.0);
	}
	bentNormal = rayBits == 0xFFFFFFFFu ? normal : tangentSpaceTransform * normalize(bentNormal);

	#if CORTAO_OUTPUT_BENT_NORMAL
		outColor = vec4(bentNormal * 0.5 + 0.5, occlusion);
	#else
		outColor = vec4(vec3(occlusion), 1.0);
	#endif
}`;
// CORTAOFilterPass.h, occlusion only.
// PORT: the reference filters one 2x2 quad per compute thread using gathers.
// Each fragment here reads the same texels for its own pixel, including the
// depth slopes taken from its 2x2 quad, so the output matches.
const FILTER_FRAGMENT = /* glsl */ `
${BASE}
uniform usampler2D tPacked;
uniform sampler2D tOcclusion;
uniform ivec2 resolution;
uniform float depthSensitivity;
uniform float normalSensitivity;

out vec4 outColor;

// CLAMP_TO_BORDER with a border of 0.
uint loadPacked(ivec2 coords) {
	if (any(lessThan(coords, ivec2(0))) || any(greaterThanEqual(coords, resolution))) {
		return 0u;
	}
	return texelFetch(tPacked, coords, 0).r;
}

float loadOcclusion(ivec2 coords) {
	return texelFetch(tOcclusion, clamp(coords, ivec2(0), resolution - 1), 0).r;
}

void main() {
	ivec2 pixelCoords = ivec2(gl_FragCoord.xy);

	uint centerPackedData = loadPacked(pixelCoords);
	if (centerPackedData == 0u) {
		outColor = vec4(1.0);
		return;
	}

	float inverseCenterDepth = 1.0 / cortao_unpackDepth(centerPackedData);

	// Depth slopes from the pixel's 2x2 quad (even pixel minus odd pixel).
	ivec2 quad = pixelCoords & ~1;
	float deltaX = cortao_unpackDepth(loadPacked(ivec2(quad.x, pixelCoords.y)))
		- cortao_unpackDepth(loadPacked(ivec2(quad.x + 1, pixelCoords.y)));
	float deltaY = cortao_unpackDepth(loadPacked(ivec2(pixelCoords.x, quad.y)))
		- cortao_unpackDepth(loadPacked(ivec2(pixelCoords.x, quad.y + 1)));
	deltaX *= inverseCenterDepth * depthSensitivity;
	deltaY *= inverseCenterDepth * depthSensitivity;

	ivec2 centerNormal = cortao_unpackNormalXY(centerPackedData);
	float normalErrorMul = normalSensitivity / float(1 << (CORTAO_PACKED_NORMAL_BITS / 2));

	#if CORTAO_LARGE_FILTER
		const int sampleCount = 6;
		ivec2 offsets[6] = ivec2[6](ivec2(3, 0), ivec2(6, 0), ivec2(3, 3), ivec2(0, 3), ivec2(0, 6), ivec2(3, -3));
	#else
		const int sampleCount = 2;
		ivec2 offsets[2] = ivec2[2](ivec2(3, 0), ivec2(0, 3));
	#endif

	float occlusionSum = loadOcclusion(pixelCoords);
	float weightSum = 1.0;

	for (int i = 0; i < sampleCount; i++) {
		float weights[2];
		for (int side = 0; side < 2; side++) {
			ivec2 offset = side == 0 ? -offsets[i] : offsets[i];
			uint packedData = loadPacked(pixelCoords + offset);

			float sampleDepth = cortao_unpackDepth(packedData) * depthSensitivity * inverseCenterDepth - depthSensitivity;
			float error = abs(deltaX * float(offset.x) + deltaY * float(offset.y) + sampleDepth);

			ivec2 normalDelta = cortao_unpackNormalXY(packedData) - centerNormal;
			error += float(normalDelta.x * normalDelta.x + normalDelta.y * normalDelta.y) * normalErrorMul;

			weights[side] = exp2(-error);
		}

		#if CORTAO_FILTER_SYMMETRY
			weights[0] = weights[1] = min(weights[0], weights[1]);
		#endif

		weightSum += weights[0] + weights[1];
		occlusionSum += loadOcclusion(pixelCoords - offsets[i]) * weights[0];
		occlusionSum += loadOcclusion(pixelCoords + offsets[i]) * weights[1];
	}

	outColor = vec4(vec3(occlusionSum / weightSum), 1.0);
}`;
const LOOKUP_TABLE_SIZES = {
    '24x24x16': [24, 24, 16],
    '32x32x24': [32, 32, 24],
    '64x64x32': [64, 64, 32],
};
/**
 * The occluder lookup table as an R32UI 3D texture, built on the CPU.
 * @param {{ directions?: number, radii?: number }} [options]
 */
export function createOccluderLookupTexture(options) {
    const { data, width, height, depth } = createOccluderLookupTable(options);
    return toLookupTexture(data, width, height, depth);
}
/** @param {Uint32Array} data @param {number} width @param {number} height @param {number} depth */
function toLookupTexture(data, width, height, depth) {
    const texture = new THREE.Data3DTexture(data, width, height, depth);
    texture.format = THREE.RedIntegerFormat;
    texture.type = THREE.UnsignedIntType;
    texture.internalFormat = 'R32UI';
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
    texture.unpackAlignment = 4;
    texture.needsUpdate = true;
    return texture;
}
/**
 * Loads one of the reference repo's prebuilt 32-ray R32_UINT tables (`tables/occluder`).
 * `createOccluderLookupTexture` builds the same data without a download.
 * @param {string} url
 * @param {'24x24x16' | '32x32x24' | '64x64x32'} [size]
 */
export async function loadOccluderLookupTable(url, size = '24x24x16') {
    const [width, height, depth] = LOOKUP_TABLE_SIZES[size];
    const response = await fetch(url);
    if (!response.ok)
        throw new Error(`Failed to load occluder lookup table ${url}: ${response.status}`);
    const data = new Uint32Array(await response.arrayBuffer());
    if (data.length !== width * height * depth) {
        throw new Error(`Occluder lookup table ${url} has ${data.length} entries, expected ${width * height * depth}`);
    }
    return toLookupTexture(data, width, height, depth);
}
/** Defaults from CORTAOConfigurationParameters. */
export const CORTAO_DEFAULTS = {
    sampleCount: 16,
    maxRadius: 1,
    maxScreenspaceRadiusHeightRatio: 0.2,
    mipBias: -4,
    maxSlopeCompensation: 4,
    backfaceThreshold: 0.5,
    backfaceBoost: 0.5,
    filterDepthSensitivity: 512,
    filterNormalSensitivity: 16,
    largeFilter: true,
    filterSymmetry: true,
};
function rawMaterial(fragmentShader, uniforms, defines = {}) {
    return new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: VERTEX,
        fragmentShader,
        uniforms,
        defines,
        depthTest: false,
        depthWrite: false,
        blending: THREE.NoBlending,
    });
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
    constructor(width, height, occluderLookupTable = createOccluderLookupTexture()) {
        this.width = width;
        this.height = height;
        this.params = { ...CORTAO_DEFAULTS };
        this.packedTarget = new THREE.WebGLRenderTarget(width, height, {
            format: THREE.RedIntegerFormat,
            type: THREE.UnsignedIntType,
            internalFormat: 'R32UI',
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            depthBuffer: false,
        });
        const occlusionOptions = {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            depthBuffer: false,
        };
        this.occlusionTarget = new THREE.WebGLRenderTarget(width, height, occlusionOptions);
        this.filteredTarget = new THREE.WebGLRenderTarget(width, height, occlusionOptions);
        this.prepassMaterial = rawMaterial(PREPASS_FRAGMENT, {
            tDepth: { value: null },
            tNormal: { value: null },
            depthScale: { value: 0 },
            depthOffset: { value: 0 },
            randomSeed: { value: 0 },
        });
        this.mainMaterial = rawMaterial(MAIN_FRAGMENT, {
            tPacked: { value: this.packedTarget.texture },
            tNormal: { value: null },
            tOccluderLookupTable: { value: occluderLookupTable },
            resolution: { value: new THREE.Vector2(width, height) },
            mipCount: { value: 1 },
            randomSeed: { value: 0 },
            normalBias: { value: 0 },
            radius: { value: 1 },
            unprojectParams: { value: new THREE.Vector4() },
            projectionParams: { value: new THREE.Vector4() },
            maxApparentCircleRadiusCoefficient: { value: 0 },
            mipBias: { value: 0 },
            inverseMaxSlopeCompensation: { value: 0 },
            backfaceThreshold: { value: 0 },
            backfaceBoost: { value: 0 },
        }, {
            CORTAO_SAMPLE_COUNT: this.params.sampleCount,
            CORTAO_RAY_COUNT: 32,
            CORTAO_OCCLUDER_LOOKUP_TABLE: occluderLookupTable ? 1 : 0,
            CORTAO_OUTPUT_BENT_NORMAL: 0,
        });
        this.filterMaterial = rawMaterial(FILTER_FRAGMENT, {
            tPacked: { value: this.packedTarget.texture },
            tOcclusion: { value: this.occlusionTarget.texture },
            resolution: { value: new THREE.Vector2(width, height) },
            depthSensitivity: { value: 0 },
            normalSensitivity: { value: 0 },
        }, { CORTAO_LARGE_FILTER: 1, CORTAO_FILTER_SYMMETRY: 1 });
        this.quad = new FullScreenQuad(this.prepassMaterial);
    }
    setDefine(material, name, value) {
        if (material.defines[name] !== value) {
            material.defines[name] = value;
            material.needsUpdate = true;
        }
    }
    /**
     * populatePrepassParameters, populateMainPassParameters and populateFilterPassParameters.
     * @param {THREE.PerspectiveCamera} camera
     * @param {THREE.DepthTexture} depthTexture
     * @param {THREE.Texture} normalTexture
     * @param {number} randomSeed
     */
    updateUniforms(camera, depthTexture, normalTexture, randomSeed) {
        const { width, height, params } = this;
        const near = camera.near;
        const far = camera.far;
        const prepass = this.prepassMaterial.uniforms;
        prepass.tDepth.value = depthTexture;
        prepass.tNormal.value = normalTexture;
        prepass.depthScale.value = (near - far) / (near * far);
        prepass.depthOffset.value = 1 / near;
        prepass.randomSeed.value = randomSeed;
        // three.js matrices are column-major, the same layout as the reference's [column][row].
        const projection = camera.projectionMatrix.elements;
        const inverse = camera.projectionMatrixInverse.elements;
        const projX = projection[0];
        const projY = projection[5];
        const projOffsetX = projection[8];
        const projOffsetY = projection[9];
        const invX = inverse[0];
        const invY = inverse[5];
        const invOffsetX = inverse[12];
        const invOffsetY = inverse[13];
        const main = this.mainMaterial.uniforms;
        main.tNormal.value = normalTexture;
        main.mipCount.value = Math.min(8, Math.floor(Math.log2(Math.max(width, height))) + 1);
        main.randomSeed.value = randomSeed;
        main.normalBias.value = (invX / width) * Math.SQRT2;
        main.radius.value = params.maxRadius;
        // flipY = false: texture coordinates run bottom to top, as in OpenGL.
        main.unprojectParams.value.set(2 * invX, 2 * invY, -invX + invOffsetX, -invY + invOffsetY);
        main.projectionParams.value.set(0.5 * projX, 0.5 * projY, -0.5 + 0.5 * projOffsetX, -0.5 + 0.5 * projOffsetY);
        const maxApparentCircleRadius = invY * params.maxScreenspaceRadiusHeightRatio * 2;
        main.maxApparentCircleRadiusCoefficient.value =
            maxApparentCircleRadius / Math.sqrt(maxApparentCircleRadius * maxApparentCircleRadius + 1);
        main.mipBias.value = Math.log2(projY * height * 0.5) + params.mipBias;
        main.inverseMaxSlopeCompensation.value = 1 / params.maxSlopeCompensation;
        main.backfaceThreshold.value = params.backfaceThreshold;
        main.backfaceBoost.value = params.backfaceBoost;
        this.setDefine(this.mainMaterial, 'CORTAO_SAMPLE_COUNT', params.sampleCount);
        const filter = this.filterMaterial.uniforms;
        filter.depthSensitivity.value = params.filterDepthSensitivity;
        filter.normalSensitivity.value = params.filterNormalSensitivity;
        this.setDefine(this.filterMaterial, 'CORTAO_LARGE_FILTER', params.largeFilter ? 1 : 0);
        this.setDefine(this.filterMaterial, 'CORTAO_FILTER_SYMMETRY', params.filterSymmetry ? 1 : 0);
    }
    renderMaterial(renderer, material, target) {
        this.quad.material = material;
        renderer.setRenderTarget(target);
        this.quad.render(renderer);
    }
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
    render(renderer, camera, depthTexture, normalTexture, { randomSeed = 0, filter = true } = {}) {
        const previousTarget = renderer.getRenderTarget();
        this.updateUniforms(camera, depthTexture, normalTexture, randomSeed >>> 0);
        this.renderMaterial(renderer, this.prepassMaterial, this.packedTarget);
        this.renderMaterial(renderer, this.mainMaterial, this.occlusionTarget);
        if (filter)
            this.renderMaterial(renderer, this.filterMaterial, this.filteredTarget);
        renderer.setRenderTarget(previousTarget);
        return filter ? this.filteredTarget.texture : this.occlusionTarget.texture;
    }
    dispose() {
        this.packedTarget.dispose();
        this.occlusionTarget.dispose();
        this.filteredTarget.dispose();
        this.prepassMaterial.dispose();
        this.mainMaterial.dispose();
        this.filterMaterial.dispose();
        this.quad.dispose();
    }
}
