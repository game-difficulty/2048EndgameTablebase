export function mountApplication(mount) {
  const recovery = window.__APP_RECOVERY__;
  const finish = () => {
    mount();
    recovery?.mounted();
  };
  const fail = () => recovery?.fail('shell_error');
  try {
    if (window.__loadRenderCompatShell) {
      window.__loadRenderCompatShell().then(finish).catch(fail);
    } else {
      finish();
    }
  } catch (error) {
    fail();
    throw error;
  }
}
