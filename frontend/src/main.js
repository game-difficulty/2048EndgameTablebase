import { createApp } from 'vue';

import App from './App.vue';
import i18n from './app/i18n';
import './style.css';
import { mountApplication } from './utils/mountApplication.js';

const app = createApp(App);
app.use(i18n);
mountApplication(() => app.mount('#app'));
