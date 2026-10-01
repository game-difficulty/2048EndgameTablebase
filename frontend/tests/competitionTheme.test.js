import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileScript, compileTemplate } from '../node_modules/@vue/compiler-sfc/dist/compiler-sfc.esm-browser.js';
const read = path => readFileSync(new URL(path, import.meta.url), 'utf8');
test('broadcast surfaces share theme variables, including stream boards and roster HUD', () => {
  for (const file of ['CompetitionMatchContent.vue','CompetitionRosterHud.vue','StreamProjectView.vue']) {
    const source = read('../src/live/content/'+file);
    assert.match(source, /var\(--match-/);
    const { descriptor, errors } = parse(source);
    assert.deepEqual(errors, []);
    const script = compileScript(descriptor,{id:'theme'});
    assert.deepEqual(compileTemplate({source:descriptor.template.content,filename:file,id:'theme',compilerOptions:{bindingMetadata:script.bindings}}).errors,[]);
  }
  const palette=read('../src/live/content/competitionTheme.css');
  assert.match(palette,/\[data-theme="light"\] \.competition-content/);
  for(const token of ['bg','card','text','copy','muted','line','cell','accent','success','hud-fade']) {
    assert.equal((palette.match(new RegExp(`--match-${token}:`,'g'))||[]).length,2,token);
  }
});
test('icons and settlement are no longer forced to dark mode', () => {
  const source=read('../src/live/content/CompetitionMatchContent.vue');
  assert.match(source,/:dark="darkTheme"/);
  assert.match(source,/darkTheme\.value\?'dark':'light'/);
  assert.doesNotMatch(source,/class="live-settlement" dark/);
});
