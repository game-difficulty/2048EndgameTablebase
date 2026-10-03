import { onBeforeUnmount, ref } from "vue";
import { uploadAsset } from "./api";

// Upload only explicit clipboard files. Never fetch clipboard HTML image URLs.
export function usePasteImage(identity) {
  const status = ref("");
  let alive = true;
  onBeforeUnmount(() => {
    alive = false;
  });
  async function paste(event, text) {
    const files = [...(event.clipboardData?.files || [])];
    if (!files.length) return;
    event.preventDefault();
    if (files.length !== 1 || !/^image\/(png|jpeg|webp)$/.test(files[0].type)) {
      status.value = "每次可粘贴一张 PNG、JPEG 或 WebP 图片。";
      return;
    }
    if (files[0].size > 5 * 1024 * 1024) {
      status.value = "图片最多 5 MiB。";
      return;
    }
    const owner = identity();
    status.value = "正在上传粘贴的图片…";
    try {
      const result = await uploadAsset(files[0], "image");
      if (!alive || identity() !== owner) return;
      const syntax = `\n![图片说明](/api/forum/v1/media/${result.id})\n`;
      if (text.value.length + syntax.length > 20000) {
        status.value = "图片已保存到我的附件；正文已满，请缩短内容后插入。";
        return;
      }
      // Append to the latest text, preserving edits made during the upload.
      text.value += syntax;
      status.value = "图片已插入正文末尾。";
    } catch (error) {
      if (alive && identity() === owner) status.value = error.message;
    }
  }
  return { paste, status };
}
