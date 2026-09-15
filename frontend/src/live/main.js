import { createApp } from "vue";
import i18n from "../app/i18n";
import "../style.css";
import LivePage from "./LivePage.vue";
import { liveBoardPalette } from "./tilePalette.js";

document.body.classList.add('live-document');
for (const [name, value] of Object.entries(liveBoardPalette())) {
  document.documentElement.style.setProperty(name, value);
}
createApp(LivePage).use(i18n).mount("#live-app");
