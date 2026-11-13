// CORTAO on WebGPURenderer: a scene pass, the CortaoNode on its depth, and the
// ambient occlusion multiplied into the color. GPU time per frame comes from
// timestamp queries, in milliseconds.

import { CortaoNode } from 'cortao.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { pass, vec4 } from 'three/tsl';
import * as THREE from 'three/webgpu';
import { SCENES } from './scenes.js';

const renderer = new THREE.WebGPURenderer({ antialias: false, trackTimestamp: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.append(renderer.domElement);
await renderer.init();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xb8c4d0);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8070, 2.2));

// Sponza when `pnpm fetch-sponza` has run, the simple scene otherwise.
const loaded = await SCENES.sponza.build().then(
	(geometry) => ({ name: 'sponza', geometry }),
	() => ({ name: 'simple', geometry: SCENES.simple.build() }),
);
const config = SCENES[loaded.name];
scene.add(new THREE.Mesh(loaded.geometry, new THREE.MeshLambertMaterial({ color: 0xd8d2c8 })));

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 200);
camera.position.fromArray(config.position);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.fromArray(config.target);
controls.update();

const scenePass = pass(scene, camera);
const cortao = new CortaoNode(scenePass.getTextureNode('depth'), () => camera);
cortao.radius.value = config.radius;
const ao = cortao.getTextureNode().r;
const color = scenePass.getTextureNode().rgb;
const pipeline = new THREE.RenderPipeline(renderer);
const outputs = {
	shaded: vec4(color.mul(ao), 1),
	aoOnly: vec4(ao, ao, ao, 1),
	off: vec4(color, 1),
};
let aoOnly = false;
const setOutput = () => {
	pipeline.outputNode = cortao.enabled ? (aoOnly ? outputs.aoOnly : outputs.shaded) : outputs.off;
	pipeline.needsUpdate = true;
};
setOutput();

const toggle = document.getElementById('toggle');
toggle.addEventListener('click', () => {
	cortao.enabled = !cortao.enabled;
	toggle.textContent = cortao.enabled ? 'AO on' : 'AO off';
	toggle.setAttribute('aria-pressed', String(cortao.enabled));
	setOutput();
});
document.getElementById('view').addEventListener('click', (event) => {
	aoOnly = !aoOnly;
	event.currentTarget.setAttribute('aria-pressed', String(aoOnly));
	setOutput();
});
const sampleButtons = document.querySelectorAll('[data-samples]');
for (const button of sampleButtons) {
	button.addEventListener('click', () => {
		cortao.sampleCount = Number(button.dataset.samples);
		for (const other of sampleButtons) other.setAttribute('aria-pressed', String(other === button));
	});
}

addEventListener('resize', () => {
	// A hidden tab reports zero, and WebGPU cannot make a zero-size swapchain.
	if (innerWidth === 0 || innerHeight === 0) return;
	camera.aspect = innerWidth / innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(innerWidth, innerHeight);
});

// GPU milliseconds per frame from timestamp queries, every 30 frames.
// resolveTimestampsAsync returns the last frame only, so this averages the
// per-pass durations the query pool keeps for every frame since the last resolve.
const timing = document.getElementById('timing');
let framesSinceResolve = 0;
let resolving = false;
async function resolveTiming() {
	resolving = true;
	framesSinceResolve = 0;
	const last = await renderer.resolveTimestampsAsync(THREE.TimestampQuery.RENDER);
	const pool = renderer.backend.timestampQueryPool?.render;
	let total = 0;
	for (const duration of pool?.timestamps?.values() ?? []) total += duration;
	const frames = pool?.frames?.length ?? 0;
	const ms = frames > 0 ? total / frames : last;
	timing.textContent =
		typeof ms === 'number'
			? `GPU ${ms.toFixed(2)} ms/frame · ${cortao.enabled ? `${cortao.sampleCount} samples` : 'AO off'}`
			: 'GPU timing unavailable (needs the timestamp-query feature)';
	resolving = false;
}

renderer.setAnimationLoop(() => {
	controls.update();
	pipeline.render();
	framesSinceResolve++;
	if (framesSinceResolve >= 30 && !resolving) void resolveTiming();
});

window.demo = { renderer, cortao, camera };
