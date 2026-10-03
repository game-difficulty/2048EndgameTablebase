<template>
  <div ref="root" class="human-account-control" data-account-menu>
    <button class="human-account-trigger" type="button" :aria-expanded="open" :aria-label="$t('auth.account.title')" @click="toggle">
      <AccountAvatar :user="user" :supporter="supporter" />
      <span>{{ displayName }}</span>
    </button>
    <AccountPopover :open="open" :auth-user="user" :account-display-name="displayName"
      :has-supporter-presentation="supporter" :can-open-admin="admin"
      @avatar="show('avatar')" @name="show('name')" @quota="show('quota')" @sponsor="show('sponsor')"
      @admin="show('admin')" @security="security" @logout="signOut" />
  </div>
  <Teleport to="body">
    <div v-if="panel" class="human-account-dialogs" @keydown.esc="panel = ''">
      <AvatarEditorDialog v-if="panel === 'avatar'" open :user="user" :supporter="supporter" @close="panel = ''" @saved="saved" />
      <DisplayNameEditorDialog v-if="panel === 'name'" open :user="user" @close="panel = ''" @saved="saved" />
      <AccountSecurityDialog v-if="panel === 'security'" open :mode="securityMode" @close="panel = ''" @success="refresh" @deactivated="refresh" />
      <SponsorDialog v-if="panel === 'sponsor'" open :user="user" @close="panel = ''" />
      <QuotaGuideDialog v-if="panel === 'quota'" open @close="panel = ''" />
      <section v-if="panel === 'admin' && admin" class="human-admin-panel">
        <button class="action-btn-small" type="button" @click="panel = ''">{{ $t('common.close') }}</button>
        <AdminPage active />
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, defineAsyncComponent, onMounted, onUnmounted, ref } from 'vue';
import AccountAvatar from '../features/auth/AccountAvatar.vue';
import AccountPopover from '../features/auth/AccountPopover.vue';

const AvatarEditorDialog = defineAsyncComponent(() => import('../features/auth/AvatarEditorDialog.vue'));
const DisplayNameEditorDialog = defineAsyncComponent(() => import('../features/auth/DisplayNameEditorDialog.vue'));
const AccountSecurityDialog = defineAsyncComponent(() => import('../features/auth/AccountSecurityDialog.vue'));
const SponsorDialog = defineAsyncComponent(() => import('../features/billing/SponsorDialog.vue'));
const QuotaGuideDialog = defineAsyncComponent(() => import('../features/billing/QuotaGuideDialog.vue'));
const AdminPage = defineAsyncComponent(() => import('../features/admin/pages/AdminPage.vue'));

const props = defineProps({ user: { type: Object, required: true } });
const emit = defineEmits(['saved', 'refresh', 'logout']);
const root = ref(null), open = ref(false), panel = ref(''), securityMode = ref('changePassword');
const displayName = computed(() => props.user.display_name || props.user.email || '');
const admin = computed(() => Boolean(props.user.management?.moderate));
const supporter = computed(() => props.user.entitlements?.tier === 'supporter' || props.user.management?.owner);

function toggle() { open.value = !open.value; if (open.value) emit('refresh'); }
function show(name) { open.value = false; panel.value = name; }
function security(mode) { securityMode.value = mode; show('security'); }
function saved(user) { panel.value = ''; emit('saved', user); }
function refresh() { panel.value = ''; emit('refresh'); }
function signOut() { open.value = false; emit('logout'); }
function outside(event) { if (!root.value?.contains(event.target)) open.value = false; }
function escape(event) { if (event.key === 'Escape') open.value = false; }

onMounted(() => { document.addEventListener('pointerdown', outside, true); document.addEventListener('keydown', escape); });
onUnmounted(() => { document.removeEventListener('pointerdown', outside, true); document.removeEventListener('keydown', escape); });
</script>
