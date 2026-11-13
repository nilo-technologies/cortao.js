// Runs the comparison in headless Chromium on the GPU and writes the PNGs and
// metrics into out/. Start the dev server first: pnpm dev
//
//   pnpm capture -- [sponza] [simple]

import { chromium } from 'playwright';

const scenes = process.argv.slice(2);
const browser = await chromium.launch({
	channel: 'chromium',
	args: ['--enable-gpu', '--use-gl=angle', '--use-angle=metal', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1320, height: 860 } });
page.on('console', (message) => {
	if (message.type() === 'error' || message.type() === 'warning') console.log(`[${message.type()}]`, message.text().slice(0, 1500));
});
page.on('pageerror', (error) => console.log('[pageerror]', error.message));

await page.goto('http://127.0.0.1:4791/');
await page.waitForFunction(() => window.lab?.status === 'ready' || window.lab?.error, null, { timeout: 120_000 });

void page.evaluate((names) => {
	void window.lab.capture(names.length > 0 ? names : undefined);
}, scenes);

let lastStatus = '';
for (;;) {
	const state = await page.evaluate(() => ({ status: window.lab.status, error: window.lab.error }));
	if (state.status !== lastStatus) {
		lastStatus = state.status;
		console.log(state.status);
	}
	if (state.error) {
		console.log(state.error);
		process.exitCode = 1;
		break;
	}
	if (state.status === 'capture done') break;
	await new Promise((resolve) => setTimeout(resolve, 500));
}

const metrics = await page.evaluate(() => window.lab.metrics);
for (const scene of Object.values(metrics)) {
	console.log(`\n${scene.scene} (${scene.renderer})`);
	for (const result of scene.results) {
		console.log(
			`${result.key.padEnd(22)} mse ${result.mse.toFixed(5)}  mean ${result.mean.toFixed(3)}  ${result.ms === undefined ? '' : `${result.ms.toFixed(2)} ms`}`,
		);
	}
}

await browser.close();
