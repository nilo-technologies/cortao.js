// The two test scenes, as single merged geometries with positions and normals only.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

function prepareGeometry(geometry, matrix) {
	const prepared = geometry.index ? geometry.clone() : geometry.clone().toNonIndexed();
	for (const name of Object.keys(prepared.attributes)) {
		if (name !== 'position' && name !== 'normal') prepared.deleteAttribute(name);
	}
	if (!prepared.index) {
		const count = prepared.attributes.position.count;
		prepared.setIndex([...Array(count).keys()]);
	}
	if (!prepared.attributes.normal) prepared.computeVertexNormals();
	prepared.applyMatrix4(matrix);
	return prepared;
}

/** Needs `pnpm fetch-sponza` first. */
export async function buildSponza() {
	const gltf = await new GLTFLoader().loadAsync(`${import.meta.env.BASE_URL}assets/sponza/Sponza.gltf`);
	gltf.scene.updateMatrixWorld(true);
	const geometries = [];
	gltf.scene.traverse((object) => {
		if (object.isMesh) geometries.push(prepareGeometry(object.geometry, object.matrixWorld));
	});
	return mergeGeometries(geometries);
}

export function buildSimple() {
	const geometries = [];
	const add = (geometry, x, y, z, rotationY = 0, rotationX = 0) => {
		const matrix = new THREE.Matrix4().compose(
			new THREE.Vector3(x, y, z),
			new THREE.Quaternion().setFromEuler(new THREE.Euler(rotationX, rotationY, 0)),
			new THREE.Vector3(1, 1, 1),
		);
		geometries.push(prepareGeometry(geometry, matrix));
	};

	add(new THREE.BoxGeometry(40, 1, 40), 0, -0.5, 0);
	add(new THREE.BoxGeometry(40, 8, 1), 0, 4, -6.5);
	for (let i = 0; i < 5; i++) add(new THREE.CylinderGeometry(0.35, 0.35, 5, 48), -6 + i * 3, 2.5, -4);
	add(new THREE.BoxGeometry(14, 0.5, 1.4), 0, 5.25, -4);

	add(new THREE.BoxGeometry(2, 2, 2), -3.2, 1, -0.5, 0.4);
	add(new THREE.BoxGeometry(1.2, 1.2, 1.2), -3.1, 2.6, -0.4, 1.1);
	add(new THREE.BoxGeometry(1, 1, 1), -1.5, 0.5, 0.6, 0.2);
	add(new THREE.SphereGeometry(1.1, 64, 48), 0.9, 1.1, -0.6);
	add(new THREE.SphereGeometry(0.45, 48, 32), 2.3, 0.45, 0.5);
	add(new THREE.TorusKnotGeometry(0.75, 0.24, 220, 32), 3.9, 1.5, -0.8, 0.5);
	add(new THREE.TorusGeometry(0.8, 0.25, 32, 96), -0.2, 0.25, 2.2, 0, Math.PI / 2);

	// Steps.
	for (let i = 0; i < 6; i++) add(new THREE.BoxGeometry(2.4, 0.25, 0.45), 6, 0.125 + i * 0.25, 1.6 - i * 0.45, -0.5);

	// Thin poles in front of the wall: the depth discontinuities horizon methods struggle with.
	for (let i = 0; i < 14; i++) add(new THREE.BoxGeometry(0.06, 1.6, 0.06), -5.2 + i * 0.32, 0.8, 2.6);
	add(new THREE.BoxGeometry(4.4, 0.06, 0.06), -3.1, 1.6, 2.6);

	return mergeGeometries(geometries);
}

export const SCENES = {
	sponza: {
		label: 'Sponza',
		build: buildSponza,
		position: [-9.5, 1.9, -0.5],
		target: [2, 3.6, 1.2],
		radius: 1,
		crop: [300, 330, 320, 180],
	},
	simple: {
		label: 'Simple scene',
		build: buildSimple,
		position: [0.2, 2.5, 6.4],
		target: [0.6, 0.9, 0],
		radius: 1,
		crop: [150, 380, 320, 180],
	},
};
