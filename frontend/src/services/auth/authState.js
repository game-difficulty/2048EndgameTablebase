import { computed, ref } from 'vue';

import { authClient } from './authClient';
import { emitAuthChanged } from './authEvents';

const ready = ref(false);
const user = ref(null);
const dialogOpen = ref(false);
const dialogMode = ref('login');
let authGeneration = 0;
let latestRefreshId = 0;

const isAuthenticated = computed(() => !!user.value);

const applyTokenBalance = (tokenBalance) => {
  if (!user.value || !tokenBalance) {
    return;
  }
  user.value = {
    ...user.value,
    token_balance: tokenBalance,
  };
};

if (typeof window !== 'undefined') {
  window.addEventListener('token-balance-updated', (event) => {
    applyTokenBalance(event?.detail || null);
  });
}

export function useAuthState() {
  const refreshAuth = async () => {
    const previousUserId = user.value?.id ?? null;
    const requestId = latestRefreshId + 1;
    latestRefreshId = requestId;
    const generationAtStart = authGeneration;
    try {
      const result = await authClient.me();
      if (requestId !== latestRefreshId || generationAtStart !== authGeneration) {
        return user.value;
      }
      user.value = result?.authenticated ? result.user : null;
      return user.value;
    } catch (error) {
      if (requestId !== latestRefreshId || generationAtStart !== authGeneration) {
        return user.value;
      }
      user.value = null;
      return null;
    } finally {
      if (requestId !== latestRefreshId || generationAtStart !== authGeneration) {
        return;
      }
      ready.value = true;
      const nextUserId = user.value?.id ?? null;
      if (previousUserId !== nextUserId) {
        emitAuthChanged();
      }
    }
  };

  const openAuthDialog = (mode = 'login') => {
    dialogMode.value = ['login', 'register', 'forgot', 'reset'].includes(mode) ? mode : 'login';
    dialogOpen.value = true;
  };

  const closeAuthDialog = () => {
    dialogOpen.value = false;
  };

  const setAuthenticatedUser = (authenticatedUser) => {
    const previousUserId = user.value?.id ?? null;
    authGeneration += 1;
    latestRefreshId += 1;
    user.value = authenticatedUser || null;
    ready.value = true;
    const nextUserId = user.value?.id ?? null;
    if (previousUserId !== nextUserId) {
      emitAuthChanged();
    }
  };

  const requireAuth = () => {
    if (user.value) {
      return true;
    }
    openAuthDialog('login');
    return false;
  };

  const logout = async () => {
    try {
      await authClient.logout();
    } finally {
      authGeneration += 1;
      latestRefreshId += 1;
      user.value = null;
      emitAuthChanged();
    }
  };

  return {
    ready,
    user,
    isAuthenticated,
    dialogOpen,
    dialogMode,
    refreshAuth,
    openAuthDialog,
    closeAuthDialog,
    setAuthenticatedUser,
    requireAuth,
    logout,
    applyTokenBalance,
  };
}
