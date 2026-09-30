import { createApp } from 'vue';
import App from './App.vue';
import './styles.css';
import { t, syncAccountLanguage } from './i18n.js';
import { api } from './api.js';

const app=createApp(App);
app.config.globalProperties.$t=t;
app.mount('#app');
void syncAccountLanguage(api.preferredLanguage);
window.addEventListener('focus',()=>{void syncAccountLanguage();});
