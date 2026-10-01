import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const read = path => readFileSync(new URL(path, import.meta.url), 'utf8');
test('live rule indicators stay between performance and time in a single fixed-height row', () => {
  const source=read('../src/live/content/StreamProjectView.vue');
  const template=source.slice(0,source.indexOf('</template>'));
  assert.ok(template.indexOf('metric.value') < template.indexOf('v-for="item in ruleMetrics"'));
  assert.ok(template.indexOf('v-for="item in ruleMetrics"') < template.indexOf('class="time-metric"'));
  assert.match(source,/grid-template-rows:48px minmax\(0,1fr\) 22px/);
  assert.match(source,/grid-template-columns:repeat\(var\(--metric-count,2\)/);
  assert.match(source,/header \.rule-metric\{text-align:center\}/);
});
test('competition player view uses the same metric ordering', () => {
  const source=read('../../competition/frontend/src/App.vue');
  const header=source.slice(source.indexOf('<header class="project-metrics"'),source.indexOf('</header>',source.indexOf('<header class="project-metrics"')));
  assert.ok(header.indexOf('performance')<header.indexOf('v-for="item in ruleMetrics(side)"'));
  assert.ok(header.indexOf('v-for="item in ruleMetrics(side)"')<header.indexOf('project-metric time'));
});
