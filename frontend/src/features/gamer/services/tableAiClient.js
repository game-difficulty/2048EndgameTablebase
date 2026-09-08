import { emitAuthRequired, emitTokenRequired } from '../../../services/auth/authEvents.js';
import { createWsClient } from '../../../services/ws/createWsClient.js';
import { createTableAiStreamTransport } from './tableAiStreamTransport.js';
import { createRandomXoshiroState } from '../../../utils/xoshiro128.js';

export function createTableAiStreamClient() {
  return createTableAiStreamTransport({
    createClient: (callbacks) => createWsClient({ ...callbacks,
      clientId: `gamer-ai-${createRandomXoshiroState().join('-')}` }),
    onFailure: (item) => {
      if (item.status === 401) emitAuthRequired();
      if (item.status === 402) emitTokenRequired(item.detail || {});
    },
  });
}
