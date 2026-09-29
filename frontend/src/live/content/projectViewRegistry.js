import Project2048View from './Project2048View.vue';
import CargoTransportView from './CargoTransportView.vue';
import ProjectPolyominoView from './ProjectPolyominoView.vue';

const renderers = Object.freeze({
  '2048-board|2048-board-v1': Project2048View,
  '2048-board|2048-board-v2': Project2048View,
  'cargo-transport|cargo-transport-v1': CargoTransportView,
  'polyomino-board|polyomino-board-v1': ProjectPolyominoView,
});

export function projectViewRenderer(view) {
  if (!view) return null;
  return renderers[`${view.view_kind}|${view.view_protocol}`] || null;
}
