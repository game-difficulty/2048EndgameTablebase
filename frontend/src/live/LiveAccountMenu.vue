<template>
  <div ref="root" data-account-menu class="live-account">
    <button class="account-trigger" :aria-expanded="open" :aria-label="$t('auth.account.title')" @click="toggle">
      <AccountAvatar :user="user" :supporter="supporter" />
      <span>{{ displayName }}</span>
    </button>
    <AccountPopover :open="open" :auth-user="user" :account-display-name="displayName"
      :has-supporter-presentation="supporter" :can-open-admin="admin"
      @avatar="show('avatar')" @name="show('name')" @quota="show('quota')" @sponsor="show('sponsor')"
      @admin="show('admin')" @security="security" @logout="signOut" />
  </div>
  <Teleport to="body">
    <div v-if="panel" class="live-account-dialogs" @keydown.esc="panel = ''">
      <AvatarEditorDialog v-if="panel === 'avatar'" open :user="user" :supporter="supporter" @close="panel = ''" @saved="saved" />
      <DisplayNameEditorDialog v-if="panel === 'name'" open :user="user" @close="panel = ''" @saved="saved" />
      <AccountSecurityDialog v-if="panel === 'security'" open :mode="securityMode" @close="panel = ''" @success="refresh" @deactivated="refresh" />
      <SponsorDialog v-if="panel === 'sponsor'" open :user="user" @close="panel = ''" />
      <QuotaGuideDialog v-if="panel === 'quota'" open @close="panel = ''" />
      <section v-if="panel === 'admin' && admin" class="live-admin-panel">
        <button class="action-btn-small" @click="panel = ''">{{ $t('common.close') }}</button>
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
const admin = computed(() => String(props.user.email || '').trim().toLowerCase() === 'assweeass@163.com' || String(props.user.display_name || '').trim().toLowerCase() === 'user0');
const supporter = computed(() => props.user.entitlements?.tier === 'supporter' || admin.value);
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

<style scoped>
.account-trigger { display:flex;align-items:center;gap:8px;padding:4px 8px;border:1px solid var(--border-main);border-radius:6px;color:var(--text-main);background:var(--bg-card); }
.account-trigger > span { max-width:9rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }
.live-account-dialogs { position:fixed;inset:0;z-index:400; }
.live-admin-panel { position:absolute;inset:24px;background:var(--bg-main);padding:16px;overflow:auto; }
</style>
