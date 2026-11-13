// Ground truth for the comparison: ambient occlusion ray traced against the
// real scene triangles (three-mesh-bvh), with the same definition CORTAO
// approximates. Cosine-distributed rays leave each visible pixel, and a ray is
// occluded when it hits geometry closer than the occlusion radius.

import { VERTEX } from 'cortao.js/webgl';
import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { BVHShaderGLSL, MeshBVHUniformStruct } from 'three-mesh-bvh';

const FRAGMENT = /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
precision highp isampler2D;

${BVHShaderGLSL.common_functions}
${BVHShaderGLSL.bvh_struct_definitions}
${BVHShaderGLSL.bvh_ray_functions}

uniform BVH bvh;
uniform sampler2D tDepth;
uniform sampler2D tNormal;
uniform ivec2 resolution;
uniform mat4 cameraWorldMatrix;
uniform vec4 unprojectParams;
uniform float depthScale;
uniform float depthOffset;
uniform float normalBias;
uniform float radius;
uniform float maxApparentCircleRadiusCoefficient;
uniform uint frameSeed;
uniform float weight;

out vec4 outColor;

uint pcg(uint v) {
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

float random(inout uint state) {
	state = pcg(state);
	return float(state >> 8u) * (1.0 / 16777216.0);
}

void main() {
	ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
	float depth = texelFetch(tDepth, pixelCoords, 0).r;
	if (depth == 1.0) {
		outColor = vec4(weight);
		return;
	}

	vec2 texCoords = (vec2(pixelCoords) + 0.5) / vec2(resolution);
	float linearDepth = 1.0 / (depth * depthScale + depthOffset);
	vec3 viewNormal = normalize(texelFetch(tNormal, pixelCoords, 0).xyz * 2.0 - 1.0);
	vec3 viewPosition = vec3(texCoords * unprojectParams.xy + unprojectParams.zw, -1.0) * linearDepth;
	viewPosition += viewNormal * (normalBias * linearDepth);

	// The same per-pixel radius CORTAO uses, including its screen-space limit.
	float clampedRadius = min(radius, linearDepth * maxApparentCircleRadiusCoefficient);

	vec3 origin = (cameraWorldMatrix * vec4(viewPosition, 1.0)).xyz;
	vec3 normal = normalize(mat3(cameraWorldMatrix) * viewNormal);
	vec3 tangent = normalize(abs(normal.y) < 0.99 ? cross(normal, vec3(0.0, 1.0, 0.0)) : cross(normal, vec3(1.0, 0.0, 0.0)));
	vec3 bitangent = cross(normal, tangent);

	uint state = pcg(uint(pixelCoords.x) + pcg(uint(pixelCoords.y) + pcg(frameSeed)));

	float unoccluded = 0.0;
	for (int i = 0; i < RAY_COUNT; i++) {
		float u = random(state);
		float phi = random(state) * 6.283185307179586;
		float r = sqrt(u);
		vec3 direction = normalize(tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + normal * sqrt(max(1.0 - u, 0.0)));

		uvec4 faceIndices = uvec4(0u);
		vec3 faceNormal = vec3(0.0);
		vec3 barycoord = vec3(0.0);
		float side = 1.0;
		float dist = 0.0;
		bool hit = bvhIntersectFirstHit(bvh, origin, direction, faceIndices, faceNormal, barycoord, side, dist);
		unoccluded += hit && dist < clampedRadius ? 0.0 : 1.0;
	}

	outColor = vec4(unoccluded / float(RAY_COUNT) * weight);
}`;

export class ReferencePass {
	constructor(width, height, meshBVH, raysPerFrame = 16) {
		this.width = width;
		this.height = height;
		this.target = new THREE.WebGLRenderTarget(width, height, {
			type: THREE.FloatType,
			minFilter: THREE.NearestFilter,
			magFilter: THREE.NearestFilter,
			depthBuffer: false,
		});

		const bvh = new MeshBVHUniformStruct();
		bvh.updateFrom(meshBVH);

		this.material = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			vertexShader: VERTEX,
			fragmentShader: FRAGMENT,
			defines: { RAY_COUNT: raysPerFrame },
			uniforms: {
				bvh: { value: bvh },
				tDepth: { value: null },
				tNormal: { value: null },
				resolution: { value: new THREE.Vector2(width, height) },
				cameraWorldMatrix: { value: new THREE.Matrix4() },
				unprojectParams: { value: new THREE.Vector4() },
				depthScale: { value: 0 },
				depthOffset: { value: 0 },
				normalBias: { value: 0 },
				radius: { value: 1 },
				maxApparentCircleRadiusCoefficient: { value: 0 },
				frameSeed: { value: 0 },
				weight: { value: 1 },
			},
			depthTest: false,
			depthWrite: false,
			// Frames are summed into the float target.
			blending: THREE.CustomBlending,
			blendEquation: THREE.AddEquation,
			blendSrc: THREE.OneFactor,
			blendDst: THREE.OneFactor,
			blendSrcAlpha: THREE.OneFactor,
			blendDstAlpha: THREE.OneFactor,
		});
		this.quad = new FullScreenQuad(this.material);
	}

	/**
	 * Sums `frames` frames of rays into `target`. Takes the projection uniforms
	 * from a CortaoPass so both use the same radius.
	 */
	async render(renderer, camera, depthTexture, normalTexture, cortao, frames, onFrame) {
		const uniforms = this.material.uniforms;
		const main = cortao.mainMaterial.uniforms;
		const prepass = cortao.prepassMaterial.uniforms;
		uniforms.tDepth.value = depthTexture;
		uniforms.tNormal.value = normalTexture;
		uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
		uniforms.unprojectParams.value.copy(main.unprojectParams.value);
		uniforms.depthScale.value = prepass.depthScale.value;
		uniforms.depthOffset.value = prepass.depthOffset.value;
		uniforms.normalBias.value = main.normalBias.value;
		uniforms.radius.value = main.radius.value;
		uniforms.maxApparentCircleRadiusCoefficient.value = main.maxApparentCircleRadiusCoefficient.value;
		uniforms.weight.value = 1 / frames;

		const previousTarget = renderer.getRenderTarget();
		const previousAutoClear = renderer.autoClear;
		renderer.setRenderTarget(this.target);
		renderer.setClearColor(0x000000, 0);
		renderer.clear();
		for (let frame = 0; frame < frames; frame++) {
			uniforms.frameSeed.value = frame + 1;
			renderer.autoClear = false;
			renderer.setRenderTarget(this.target);
			this.quad.render(renderer);
			renderer.autoClear = previousAutoClear;
			renderer.setRenderTarget(previousTarget);
			// Lets the caller wait for the GPU, so a long trace is not one giant GPU task.
			await onFrame?.(frame);
		}
		return this.target.texture;
	}
}
