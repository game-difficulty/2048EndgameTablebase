# 下注图标彩色透明版本

## 当前填色提示词

Use case: precise-object-edit. Edit the provided transparent betting icon by adding tasteful flat color fills inside the existing shapes. Preserve the exact silhouette, composition, dark indigo-purple outlines, line widths, spade, cards, chip and dice geometry. Fill the front playing card warm ivory, the back card pale lavender, the spade solid dark indigo, the casino chip muted teal with light ivory edge segments, and the three die faces coordinated golden yellow, warm apricot and soft coral. Keep die pips dark indigo for strong contrast. Clean simple flat colors, no gradients, no new objects, no text, no glow, no cast shadow, no added border around the overall icon. This icon must read clearly at 40 pixels on both navy and cream UI backgrounds. The exterior background and gaps BETWEEN separate objects must remain genuinely transparent alpha; only the interiors of the existing objects get color. Keep the same square canvas framing and generous transparent margin.

## 资源与验证

- 来源：用户提供的 `C:/Users/Administrator/Downloads/OIP.webp`。
- 背景提取与彩色填充：内置 imagegen（非 CLI/API 模式）。
- 最终资源：`frontend/src/features/roomActivities/assets/prediction-colored.webp`；保留之前的单色透明版本。
- 打包：生成图等比缩小至 160×160、WebP 无损编码，保留 alpha；17,848 字节。
- 显示：礼物栏使用 52×52px 图像区域，抵消原图透明留白，与相邻礼物图标的视觉大小及标题行对齐。米白／淡紫扑克牌、青绿筹码、金黄／杏橙／珊瑚色骰子；深浅主题均显示原配色，移除单色 filter。
- 验证：RGBA、alpha 范围 0–255、角落 alpha 为 0；浏览器深浅主题均检查通过。

## 背景提取提示词（前一版本）

Use case: background-extraction. Edit target: the provided existing betting icon. Remove all white background and all white negative space, including the white spaces within the playing cards, chip and dice faces, producing a genuinely transparent RGBA PNG with alpha (not a checkerboard painted into the picture). Keep the exact original dark muted-purple strokes, spade, playing cards, chip and die, their geometry, relative positions, line weights, orientation, and square canvas composition unchanged. This is background removal only. No redesign, no additional elements, no white fill remaining, no shadows, no colored background. Preserve smooth antialiased edges without white halos. The result will be used at 40px as the same website UI icon.
