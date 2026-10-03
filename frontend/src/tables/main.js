import { createApp } from 'vue';
import App from '../App.vue';
import i18n from '../app/i18n';
import '../style.css';

createApp(App).use(i18n).mount('#app');
