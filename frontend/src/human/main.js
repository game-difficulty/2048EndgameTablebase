import { createApp } from 'vue';
import i18n from '../app/i18n';
import HumanApp from './HumanApp.vue';
import './human.css';

createApp(HumanApp).use(i18n).mount('#human-app');
