import { mkdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const OUT = fileURLToPath(new URL('./out/', import.meta.url));
const NAME = /^[\w.-]+\.(png|json)$/;

/** POST /save?name=<file> writes the request body to examples/out/<file>. */
function saveToOut() {
	return {
		name: 'cortao-save',
		configureServer(server) {
			server.middlewares.use('/save', (request, response) => {
				const name = new URL(request.url ?? '', 'http://localhost').searchParams.get('name') ?? '';
				if (request.method !== 'POST' || !NAME.test(name)) {
					response.statusCode = 400;
					response.end('expected POST /save?name=<file>.png|json');
					return;
				}
				const chunks = [];
				request.on('data', (chunk) => chunks.push(chunk));
				request.on('end', () => {
					mkdirSync(OUT, { recursive: true });
					writeFileSync(`${OUT}${name}`, Buffer.concat(chunks));
					response.end('ok');
				});
			});
		},
	};
}

export default defineConfig({
	plugins: [saveToOut()],
	build: {
		target: 'es2022',
		rollupOptions: {
			input: {
				compare: fileURLToPath(new URL('./index.html', import.meta.url)),
				webgpu: fileURLToPath(new URL('./webgpu.html', import.meta.url)),
			},
		},
	},
});
