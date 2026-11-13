// Comparison harness: renders one G-buffer per scene, then CORTAO, three.js's
// GTAOPass and a ray-traced reference from that same G-buffer, measures each
// against the reference and saves PNGs and metrics to examples/out/ through
// the dev server.

import { CortaoPass, VERTEX } from 'cortao.js/webgl';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { MeshBVH } from 'three-mesh-bvh';
import { ReferencePass } from './reference.js';
import { SCENES } from './scenes.js';

const WIDTH = 1280;
const HEIGHT = 720;
const TEMPORAL_FRAMES = 32;
const REFERENCE_FRAMES = 256;
const REFERENCE_RAYS_PER_FRAME = 4;

const canvas = document.getElementById('canvas');
const statusElement = document.getElementById('status');
const sceneSelect = document.getElementById('scene');
const viewSelect = document.getElementById('view');
const samplesSelect = document.getElementById('samples');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(WIDTH, HEIGHT, false);
const gl = renderer.getContext();

const camera = new THREE.PerspectiveCamera(60, WIDTH / HEIGHT, 0.1, 100);
const controls = new OrbitControls(camera, canvas);
const scene = new THREE.Scene();
const normalMaterial = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });

function setStatus(text) {
	statusElement.textContent = text;
	lab.status = text;
}

function nextFrame() {
	return new Promise((resolve) => setTimeout(resolve, 0));
}

// G-buffer: view-space normals (rgb = n * 0.5 + 0.5, alpha = 1 on geometry) and float depth.
const gbuffer = new THREE.WebGLRenderTarget(WIDTH, HEIGHT, {
	type: THREE.HalfFloatType,
	minFilter: THREE.NearestFilter,
	magFilter: THREE.NearestFilter,
	depthTexture: new THREE.DepthTexture(WIDTH, HEIGHT, THREE.FloatType),
});

function renderGBuffer() {
	camera.updateMatrixWorld();
	renderer.setRenderTarget(gbuffer);
	renderer.setClearColor(0x8080ff, 0);
	renderer.clear();
	renderer.render(scene, camera);
	renderer.setRenderTarget(null);
}

// Copies a texture to the canvas or to an 8-bit target, without any colour conversion.
const presentMaterial = new THREE.RawShaderMaterial({
	glslVersion: THREE.GLSL3,
	vertexShader: VERTEX,
	fragmentShader: /* glsl */ `
		precision highp float;
		uniform sampler2D tSource;
		uniform int mode;
		out vec4 outColor;
		void main() {
			vec4 texel = texelFetch(tSource, ivec2(gl_FragCoord.xy), 0);
			outColor = vec4(mode == 1 ? texel.rgb : mode == 2 ? texel.aaa : texel.rrr, 1.0);
		}`,
	uniforms: { tSource: { value: null }, mode: { value: 0 } },
	depthTest: false,
	depthWrite: false,
	blending: THREE.NoBlending,
});

// Sums frames into a float target for the temporal average.
const accumulateMaterial = new THREE.RawShaderMaterial({
	glslVersion: THREE.GLSL3,
	vertexShader: VERTEX,
	fragmentShader: /* glsl */ `
		precision highp float;
		uniform sampler2D tSource;
		uniform float weight;
		out vec4 outColor;
		void main() {
			outColor = texelFetch(tSource, ivec2(gl_FragCoord.xy), 0) * weight;
		}`,
	uniforms: { tSource: { value: null }, weight: { value: 1 } },
	depthTest: false,
	depthWrite: false,
	blending: THREE.CustomBlending,
	blendEquation: THREE.AddEquation,
	blendSrc: THREE.OneFactor,
	blendDst: THREE.OneFactor,
	blendSrcAlpha: THREE.OneFactor,
	blendDstAlpha: THREE.OneFactor,
});

const quad = new FullScreenQuad(presentMaterial);
const displayTarget = new THREE.WebGLRenderTarget(WIDTH, HEIGHT, { depthBuffer: false });
const accumulationTarget = new THREE.WebGLRenderTarget(WIDTH, HEIGHT, {
	type: THREE.FloatType,
	minFilter: THREE.NearestFilter,
	magFilter: THREE.NearestFilter,
	depthBuffer: false,
});

const PRESENT_MODE = { red: 0, rgb: 1, alpha: 2 };

function present(texture, target = null, mode = PRESENT_MODE.red) {
	presentMaterial.uniforms.tSource.value = texture;
	presentMaterial.uniforms.mode.value = mode;
	quad.material = presentMaterial;
	renderer.setRenderTarget(target);
	quad.render(renderer);
	renderer.setRenderTarget(null);
}

/** Reads a texture back as 8-bit RGBA, bottom row first. */
function readPixels(texture, mode = PRESENT_MODE.red) {
	present(texture, displayTarget, mode);
	const pixels = new Uint8Array(WIDTH * HEIGHT * 4);
	renderer.readRenderTargetPixels(displayTarget, 0, 0, WIDTH, HEIGHT, pixels);
	return pixels;
}

const loaded = new Map();
let current = null;
let cortao = null;
let gtao = null;

async function loadScene(name) {
	if (!loaded.has(name)) {
		setStatus(`building ${name}`);
		await nextFrame();
		const geometry = await SCENES[name].build();
		const bvh = new MeshBVH(geometry);
		const mesh = new THREE.Mesh(geometry, normalMaterial);
		const reference = new ReferencePass(WIDTH, HEIGHT, bvh, REFERENCE_RAYS_PER_FRAME);
		geometry.computeBoundingBox();
		loaded.set(name, { name, mesh, reference, triangles: geometry.index.count / 3, bounds: geometry.boundingBox });
	}
	if (current) scene.remove(current.mesh);
	current = loaded.get(name);
	scene.add(current.mesh);

	const config = SCENES[name];
	camera.position.fromArray(config.position);
	controls.target.fromArray(config.target);
	controls.update();
	cortao.params.maxRadius = config.radius;
	gtao.updateGtaoMaterial({ radius: config.radius });
}

// AO variants ----------------------------------------------------------------

function seedForFrame(frame) {
	return (Math.imul(frame + 1, 0x9e3779b1) ^ 0x5bd1e995) >>> 0;
}

function renderCortao(sampleCount, { filter, frame = 0 }) {
	cortao.params.sampleCount = sampleCount;
	return cortao.render(renderer, camera, gbuffer.depthTexture, gbuffer.texture, {
		randomSeed: seedForFrame(frame),
		filter,
	});
}

/** Stand-in for the temporal filter the reference expects after its spatial filter: a plain average over frames with a still camera. */
function renderCortaoTemporal(sampleCount, frames = TEMPORAL_FRAMES) {
	renderer.setRenderTarget(accumulationTarget);
	renderer.setClearColor(0x000000, 0);
	renderer.clear();
	const previousAutoClear = renderer.autoClear;
	renderer.autoClear = false;
	accumulateMaterial.uniforms.weight.value = 1 / frames;
	for (let frame = 0; frame < frames; frame++) {
		accumulateMaterial.uniforms.tSource.value = renderCortao(sampleCount, { filter: true, frame });
		quad.material = accumulateMaterial;
		renderer.setRenderTarget(accumulationTarget);
		quad.render(renderer);
	}
	renderer.autoClear = previousAutoClear;
	renderer.setRenderTarget(null);
	return accumulationTarget.texture;
}

function renderGtao(samples, denoised) {
	gtao.updateGtaoMaterial({ samples });
	gtao.render(renderer, null, null);
	renderer.setRenderTarget(null);
	return denoised ? gtao.pdRenderTarget.texture : gtao.gtaoRenderTarget.texture;
}

/** Blocks until the GPU has finished everything submitted so far. */
function waitForGpu() {
	renderer.readRenderTargetPixels(displayTarget, 0, 0, 1, 1, new Uint8Array(4));
}

async function renderReference(frames = REFERENCE_FRAMES) {
	cortao.updateUniforms(camera, gbuffer.depthTexture, gbuffer.texture, 0);
	let last = performance.now();
	return current.reference.render(renderer, camera, gbuffer.depthTexture, gbuffer.texture, cortao, frames, async (frame) => {
		waitForGpu();
		// Rests the GPU after every pass, for `lab.idle` times as long as the pass took.
		if (lab.idle > 0) {
			await new Promise((resolve) => setTimeout(resolve, (performance.now() - last) * lab.idle));
			last = performance.now();
		}
		if (frame % 8 === 7) {
			setStatus(`ray tracing reference ${frame + 1}/${frames}`);
			await nextFrame();
		}
	});
}

/**
 * How `measure` timed the last call: 'gpu' from EXT_disjoint_timer_query_webgl2,
 * or 'wall' (GPU drained before and after, so CPU and driver time are included).
 */
let timer = 'wall';

/** Average milliseconds per call. */
async function measure(render, iterations = 20) {
	const gl = renderer.getContext();
	const timerQuery = gl.getExtension('EXT_disjoint_timer_query_webgl2');
	const sync = (texture) => {
		present(texture, displayTarget);
		waitForGpu();
	};
	sync(render());
	if (timerQuery) {
		const query = gl.createQuery();
		gl.beginQuery(timerQuery.TIME_ELAPSED_EXT, query);
		let texture;
		for (let i = 0; i < iterations; i++) texture = render();
		gl.endQuery(timerQuery.TIME_ELAPSED_EXT);
		sync(texture);
		while (!gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) await nextFrame();
		const disjoint = gl.getParameter(timerQuery.GPU_DISJOINT_EXT);
		const nanoseconds = gl.getQueryParameter(query, gl.QUERY_RESULT);
		gl.deleteQuery(query);
		if (!disjoint) {
			timer = 'gpu';
			return nanoseconds / 1e6 / iterations;
		}
	}
	timer = 'wall';
	const start = performance.now();
	let texture;
	for (let i = 0; i < iterations; i++) texture = render();
	sync(texture);
	return (performance.now() - start) / iterations;
}

// Live view ------------------------------------------------------------------

let viewToken = 0;

async function renderView() {
	if (!current) return;
	const token = ++viewToken;
	renderGBuffer();
	const samples = Number(samplesSelect.value);
	const view = viewSelect.value;
	if (view === 'normal') present(gbuffer.texture, null, PRESENT_MODE.rgb);
	else if (view === 'cortao') present(renderCortao(samples, { filter: true }));
	else if (view === 'cortaoRaw') present(renderCortao(samples, { filter: false }));
	else if (view === 'cortaoTemporal') present(renderCortaoTemporal(samples));
	else if (view === 'gtao') present(renderGtao(samples, true));
	else if (view === 'gtaoRaw') present(renderGtao(samples, false));
	else if (view === 'reference') {
		const texture = await renderReference(32);
		if (token === viewToken) present(texture);
	}
}

// Capture --------------------------------------------------------------------

function toCanvas(pixels) {
	const target = document.createElement('canvas');
	target.width = WIDTH;
	target.height = HEIGHT;
	const context = target.getContext('2d');
	const image = context.createImageData(WIDTH, HEIGHT);
	for (let y = 0; y < HEIGHT; y++) {
		const row = pixels.subarray((HEIGHT - 1 - y) * WIDTH * 4, (HEIGHT - y) * WIDTH * 4);
		image.data.set(row, y * WIDTH * 4);
	}
	context.putImageData(image, 0, 0);
	return target;
}

async function save(name, target) {
	const blob = await new Promise((resolve) => target.toBlob(resolve, 'image/png'));
	const response = await fetch(`/save?name=${encodeURIComponent(name)}`, { method: 'POST', body: blob });
	if (!response.ok) throw new Error(`Saving ${name} failed: ${response.status}`);
}

function compare(pixels, reference, mask) {
	let squared = 0;
	let sum = 0;
	let count = 0;
	for (let i = 0; i < mask.length; i += 4) {
		if (mask[i] < 128) continue;
		const value = pixels[i] / 255;
		const difference = value - reference[i] / 255;
		squared += difference * difference;
		sum += value;
		count++;
	}
	return { mse: squared / count, mean: sum / count };
}

/** Signed error against the reference: red is too dark, blue is too bright. */
function errorPixels(pixels, reference, mask, gain = 4) {
	const output = new Uint8Array(pixels.length);
	for (let i = 0; i < pixels.length; i += 4) {
		const difference = mask[i] < 128 ? 0 : ((pixels[i] - reference[i]) / 255) * gain;
		const tooDark = Math.min(1, Math.max(0, -difference));
		const tooBright = Math.min(1, Math.max(0, difference));
		output[i] = 255 * (1 - tooBright);
		output[i + 1] = 255 * (1 - tooDark - tooBright);
		output[i + 2] = 255 * (1 - tooDark);
		output[i + 3] = 255;
	}
	return output;
}

function caption(result) {
	const parts = [];
	if (result.mse !== undefined) parts.push(`MSE ${result.mse.toFixed(5)}`);
	if (result.mean !== undefined) parts.push(`mean ${result.mean.toFixed(3)}`);
	if (result.ms !== undefined) parts.push(`${result.ms.toFixed(2)} ms ${timer}`);
	return parts.join('   ');
}

/** Lays results out as captioned tiles. `crop` is [x, y, width, height] from the top left. */
function composeGrid(results, columns, { crop = null, tileWidth = 640 } = {}) {
	const source = crop ?? [0, 0, WIDTH, HEIGHT];
	const tileHeight = Math.round((tileWidth * source[3]) / source[2]);
	const captionHeight = 46;
	const gap = 6;
	const rows = Math.ceil(results.length / columns);
	const grid = document.createElement('canvas');
	grid.width = columns * tileWidth + (columns + 1) * gap;
	grid.height = rows * (tileHeight + captionHeight) + (rows + 1) * gap;
	const context = grid.getContext('2d');
	context.fillStyle = '#16161a';
	context.fillRect(0, 0, grid.width, grid.height);
	context.imageSmoothingEnabled = !crop;
	results.forEach((result, index) => {
		const x = gap + (index % columns) * (tileWidth + gap);
		const y = gap + Math.floor(index / columns) * (tileHeight + captionHeight + gap);
		context.fillStyle = '#f2f2f2';
		context.font = '600 16px ui-monospace, Menlo, monospace';
		context.fillText(result.label, x + 4, y + 18);
		context.fillStyle = '#a8a8b0';
		context.font = '13px ui-monospace, Menlo, monospace';
		context.fillText(result.caption ?? caption(result), x + 4, y + 37);
		context.drawImage(result.canvas, ...source, x, y + captionHeight, tileWidth, tileHeight);
	});
	return grid;
}

async function captureScene(name) {
	await loadScene(name);
	renderGBuffer();
	const config = SCENES[name];
	const results = new Map();
	const mask = readPixels(gbuffer.texture, PRESENT_MODE.alpha);

	setStatus(`${name}: ray tracing reference`);
	await nextFrame();
	const referencePixels = readPixels(await renderReference());
	const add = (key, label, pixels, ms) => {
		const result = { key, label, pixels, canvas: toCanvas(pixels), ms, ...compare(pixels, referencePixels, mask) };
		results.set(key, result);
		return result;
	};
	add('reference', `Ray-traced reference, ${REFERENCE_FRAMES * REFERENCE_RAYS_PER_FRAME} rays per pixel`, referencePixels);
	results.get('reference').caption = `mean ${results.get('reference').mean.toFixed(3)}   radius ${config.radius} m`;

	for (const samples of [4, 8, 16, 32]) {
		setStatus(`${name}: CORTAO ${samples}`);
		await nextFrame();
		add(
			`cortao-${samples}-raw`,
			`CORTAO ${samples} samples, unfiltered`,
			readPixels(renderCortao(samples, { filter: false })),
			await measure(() => renderCortao(samples, { filter: false })),
		);
		add(
			`cortao-${samples}-filtered`,
			`CORTAO ${samples} samples, filtered`,
			readPixels(renderCortao(samples, { filter: true })),
			await measure(() => renderCortao(samples, { filter: true })),
		);
		add(
			`cortao-${samples}-temporal`,
			`CORTAO ${samples} samples, filtered, ${TEMPORAL_FRAMES} frames`,
			readPixels(renderCortaoTemporal(samples)),
			results.get(`cortao-${samples}-filtered`).ms,
		);
	}

	for (const samples of [8, 16, 32]) {
		setStatus(`${name}: GTAO ${samples}`);
		await nextFrame();
		const ms = await measure(() => renderGtao(samples, true));
		add(`gtao-${samples}-raw`, `three.js GTAOPass ${samples} samples, unfiltered`, readPixels(renderGtao(samples, false)), ms);
		add(`gtao-${samples}-denoised`, `three.js GTAOPass ${samples} samples, denoised`, readPixels(renderGtao(samples, true)), ms);
	}

	const normals = {
		label: 'View-space normals',
		caption: `${current.triangles} triangles`,
		canvas: toCanvas(readPixels(gbuffer.texture, PRESENT_MODE.rgb)),
	};
	const error = (key) => {
		const result = results.get(key);
		return { ...result, label: `Error: ${result.label}`, canvas: toCanvas(errorPixels(result.pixels, referencePixels, mask)) };
	};
	const pick = (...keys) => keys.map((key) => results.get(key));

	setStatus(`${name}: saving`);
	await nextFrame();
	for (const result of results.values()) await save(`${name}-${result.key}.png`, result.canvas);
	await save(`${name}-normals.png`, normals.canvas);

	await save(
		`${name}-grid-overview.png`,
		composeGrid(
			[...pick('reference', 'cortao-16-temporal', 'gtao-16-denoised'), normals, error('cortao-16-temporal'), error('gtao-16-denoised')],
			3,
		),
	);
	await save(
		`${name}-grid-cortao-samples.png`,
		composeGrid(
			pick('cortao-4-raw', 'cortao-4-filtered', 'cortao-4-temporal', 'cortao-16-raw', 'cortao-16-filtered', 'cortao-16-temporal'),
			3,
		),
	);
	await save(
		`${name}-grid-low-budget.png`,
		composeGrid(
			[...pick('reference', 'cortao-4-temporal', 'gtao-8-denoised'), normals, error('cortao-4-temporal'), error('gtao-8-denoised')],
			3,
		),
	);
	await save(
		`${name}-grid-crop.png`,
		composeGrid(pick('reference', 'cortao-4-temporal', 'cortao-16-temporal', 'gtao-16-denoised'), 4, { crop: config.crop }),
	);

	const metrics = {
		scene: config.label,
		resolution: [WIDTH, HEIGHT],
		radius: config.radius,
		triangles: current.triangles,
		timer,
		renderer: gl.getParameter(gl.getExtension('WEBGL_debug_renderer_info')?.UNMASKED_RENDERER_WEBGL ?? gl.RENDERER),
		results: [...results.values()].map(({ key, label, mse, mean, ms }) => ({ key, label, mse, mean, ms })),
	};
	await fetch(`/save?name=${name}-metrics.json`, { method: 'POST', body: JSON.stringify(metrics, null, 2) });
	return metrics;
}

// Video -------------------------------------------------------------------------

const VIDEO_PATHS = {
	// Walks down the nave while the view turns.
	sponza: (t) => ({
		position: [-9.5 + 7 * t, 1.9 + 0.6 * Math.sin(t * Math.PI), -0.5 + 0.8 * Math.sin(t * Math.PI * 2)],
		target: [2 + 6 * t, 3.6 - 1.4 * t, 1.2 - 2.4 * t],
	}),
	// Orbits the scene.
	simple: (t) => {
		const angle = -0.9 + 1.8 * t;
		return { position: [0.6 + 6.6 * Math.sin(angle), 2.5 + 0.8 * Math.sin(t * Math.PI), 6.6 * Math.cos(angle)], target: [0.6, 0.9, 0] };
	},
};

/**
 * Saves one composite frame per step of a camera move: reference, CORTAO at 16 and
 * 4 samples and GTAOPass. CORTAO takes a new seed every frame and has no temporal
 * filter here, so its frame-to-frame noise is what the spatial filter leaves.
 */
async function captureVideo(name, frames = 150, first = 0, referenceFrames = 24) {
	await loadScene(name);
	const composite = document.createElement('canvas');
	composite.width = WIDTH;
	composite.height = HEIGHT;
	const context = composite.getContext('2d');
	context.font = '600 15px ui-monospace, Menlo, monospace';
	const tile = (texture, column, row, label) => {
		const x = (column * WIDTH) / 2;
		const y = (row * HEIGHT) / 2;
		context.drawImage(toCanvas(readPixels(texture)), x, y, WIDTH / 2, HEIGHT / 2);
		context.fillStyle = 'rgba(0, 0, 0, 0.72)';
		context.fillRect(x + 8, y + 8, context.measureText(label).width + 16, 26);
		context.fillStyle = '#fff';
		context.fillText(label, x + 16, y + 26);
	};
	for (let frame = first; frame < frames; frame++) {
		const t = frame / (frames - 1);
		const eased = t * t * (3 - 2 * t);
		const view = VIDEO_PATHS[name](eased);
		camera.position.fromArray(view.position);
		camera.lookAt(...view.target);
		renderGBuffer();
		tile(await renderReference(referenceFrames), 0, 0, `Ray-traced reference, ${referenceFrames * REFERENCE_RAYS_PER_FRAME} rays`);
		tile(renderCortao(16, { filter: true, frame }), 1, 0, 'CORTAO, 16 samples');
		tile(renderGtao(16, true), 0, 1, 'GTAOPass, 16 samples');
		tile(renderCortao(4, { filter: true, frame }), 1, 1, 'CORTAO, 4 samples');
		const blob = await new Promise((resolve) => composite.toBlob(resolve, 'image/png'));
		const file = `video-${name}-${String(frame).padStart(4, '0')}.png`;
		const response = await fetch(`/save?name=${file}`, { method: 'POST', body: blob });
		if (!response.ok) throw new Error(`Saving ${file} failed: ${response.status}`);
		setStatus(`video ${name} ${frame + 1}/${frames}`);
		await nextFrame();
	}
	setStatus('video done');
}

const lab = {
	status: 'loading',
	/** GPU rest between reference passes, as a multiple of the pass time; 0 runs flat out. */
	idle: 0,
	error: null,
	metrics: {},
	renderView,
	async video(name, frames, first) {
		lab.error = null;
		try {
			await captureVideo(name, frames, first);
		} catch (error) {
			lab.error = String(error?.stack ?? error);
			setStatus(`video failed: ${error}`);
		}
	},
	camera,
	controls,
	async capture(names = Object.keys(SCENES)) {
		lab.error = null;
		try {
			for (const name of names) lab.metrics[name] = await captureScene(name);
			setStatus('capture done');
		} catch (error) {
			lab.error = String(error?.stack ?? error);
			setStatus(`capture failed: ${error}`);
		}
		renderView();
	},
};
window.lab = lab;

async function start() {
	cortao = new CortaoPass(WIDTH, HEIGHT);
	gtao = new GTAOPass(scene, camera, WIDTH, HEIGHT);
	gtao.setGBuffer(gbuffer.depthTexture, gbuffer.texture);
	gtao.output = GTAOPass.OUTPUT.Off;
	lab.cortao = cortao;
	lab.gtao = gtao;

	await loadScene(sceneSelect.value);
	renderView();
	setStatus('ready');

	controls.addEventListener('change', renderView);
	viewSelect.addEventListener('change', renderView);
	samplesSelect.addEventListener('change', renderView);
	sceneSelect.addEventListener('change', async () => {
		await loadScene(sceneSelect.value);
		renderView();
		setStatus('ready');
	});
	document.getElementById('capture').addEventListener('click', () => lab.capture([sceneSelect.value]));
}

start().catch((error) => {
	lab.error = String(error?.stack ?? error);
	setStatus(`failed: ${error}`);
	console.error(error);
});
