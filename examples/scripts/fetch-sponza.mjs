// Downloads Sponza's geometry from Khronos glTF-Sample-Assets into
// examples/public/assets/sponza/, without its textures and materials (the
// examples only need positions and normals). Sponza is not redistributed here;
// see the model's own license at
// https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza
//
//   pnpm fetch-sponza

import { mkdirSync, writeFileSync } from 'node:fs';

const SOURCE = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Sponza/glTF/';
const OUT = new URL('../public/assets/sponza/', import.meta.url);

async function download(file) {
	const response = await fetch(`${SOURCE}${file}`);
	if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
	return Buffer.from(await response.arrayBuffer());
}

const gltf = JSON.parse((await download('Sponza.gltf')).toString('utf8'));
for (const key of ['images', 'textures', 'samplers', 'materials']) delete gltf[key];
for (const mesh of gltf.meshes) for (const primitive of mesh.primitives) delete primitive.material;
if (gltf.buffers.length !== 1 || gltf.buffers[0].uri !== 'Sponza.bin') throw new Error('unexpected buffers in Sponza.gltf');

const bin = await download('Sponza.bin');
mkdirSync(OUT, { recursive: true });
writeFileSync(new URL('Sponza.gltf', OUT), JSON.stringify(gltf));
writeFileSync(new URL('Sponza.bin', OUT), bin);
console.log(`Sponza: ${gltf.meshes.length} meshes, ${(bin.length / 1e6).toFixed(1)} MB into ${OUT.pathname}`);
