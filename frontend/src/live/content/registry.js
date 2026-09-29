import ClassicAiContent from './ClassicAiContent.vue';
import MultiAiContent from './MultiAiContent.vue';
import HumanPlayContent from './HumanPlayContent.vue';
import CompetitionMatchContent from './CompetitionMatchContent.vue';
// Register each new content kind explicitly; never silently decode an unknown protocol.
export const contentRegistry = Object.freeze({ 'classic-ai': ClassicAiContent, 'classic-multi-ai': MultiAiContent, 'human-play': HumanPlayContent, 'competition-match': CompetitionMatchContent });
export const contentProtocols = Object.freeze({ 'classic-ai': 'classic-step-v1', 'classic-multi-ai': 'classic-multi-v1', 'human-play': 'human-play-v1', 'competition-match': 'competition-match-v1' });
