// Renders the camera-move comparison frames in headless Chromium and encodes them
// with ffmpeg into out/. Start the dev server first: pnpm dev
//
//   pnpm video -- [sponza] [simple]

import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';

const OUT = new URL('../out/', import.meta.url).pathname;
const FRAMES = Number(process.env.FRAMES ?? 150);
// IDLE rests the GPU between ray-tracing passes (a multiple of each pass's time), so the machine stays usable.
const IDLE = Number(process.env.IDLE ?? 3);
// FIRST resumes a stopped run at that frame.
const FIRST = Number(process.env.FIRST ?? 0);
const scenes = process.argv.slice(2).length > 0 ? process.argv.slice(2) : ['sponza', 'simple'];
const browser = await chromium.launch({
	channel: 'chromium',
	args: ['--enable-gpu', '--use-gl=angle', '--use-angle=metal', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1320, height: 860 } });
page.on('pageerror', (error) => console.log('[pageerror]', error.message));
await page.goto('http://127.0.0.1:4791/');
await page.waitForFunction(() => window.lab?.status === 'ready' || window.lab?.error, null, { timeout: 120_000 });

for (const scene of scenes) {
	void page.evaluate(
		([name, frames, first, idle]) => {
			window.lab.idle = idle;
			void window.lab.video(name, frames, first);
		},
		[scene, FRAMES, FIRST, IDLE],
	);
	let last = '';
	for (;;) {
		const state = await page.evaluate(() => ({ status: window.lab.status, error: window.lab.error }));
		if (state.error) throw new Error(state.error);
		if (state.status === 'video done') break;
		if (state.status !== last && /0\/|done/.test(state.status)) console.log(state.status);
		last = state.status;
		await new Promise((resolve) => setTimeout(resolve, 500));
	}
	await page.evaluate(() => (window.lab.status = 'ready'));
	execFileSync('ffmpeg', [
		'-y',
		'-loglevel',
		'error',
		'-framerate',
		'30',
		'-i',
		`${OUT}video-${scene}-%04d.png`,
		'-c:v',
		'libx264',
		'-pix_fmt',
		'yuv420p',
		'-crf',
		'20',
		'-movflags',
		'+faststart',
		`${OUT}${scene}-move.mp4`,
	]);
	console.log('encoded', `${scene}-move.mp4`);
}
await browser.close();
