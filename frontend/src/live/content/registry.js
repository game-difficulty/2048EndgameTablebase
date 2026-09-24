import ClassicAiContent from './ClassicAiContent.vue';
import MultiAiContent from './MultiAiContent.vue';
// Register each new content kind explicitly; never silently decode an unknown protocol.
export const contentRegistry = Object.freeze({ 'classic-ai': ClassicAiContent, 'classic-multi-ai': MultiAiContent });
export const contentProtocols = Object.freeze({ 'classic-ai': 'classic-step-v1', 'classic-multi-ai': 'classic-multi-v1' });
