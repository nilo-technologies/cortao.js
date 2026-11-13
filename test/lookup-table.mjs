// Checks the generated occluder lookup table against the reference's own
// 24x24x16 table (test/fixtures, from theagentd/CORTAO v1.0.1, MIT).
// Run after `pnpm build`: node test/lookup-table.mjs

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createOccluderLookupTable } from '../dist/lookupTable.js';

const fixture = new Uint32Array(readFileSync(new URL('./fixtures/24x24x16x32_R32_UINT.bin', import.meta.url)).buffer.slice(0));
const table = createOccluderLookupTable();

assert.equal(table.data.length, fixture.length, 'entry count');
let mismatches = 0;
for (let i = 0; i < fixture.length; i++) if (table.data[i] !== fixture[i]) mismatches++;
assert.equal(mismatches, 0, `${mismatches} of ${fixture.length} entries differ from the reference table`);

const larger = createOccluderLookupTable({ directions: 32, radii: 24 });
assert.equal(larger.data.length, 32 * 32 * 24);

console.log(`lookup table matches the reference: ${fixture.length} entries`);
