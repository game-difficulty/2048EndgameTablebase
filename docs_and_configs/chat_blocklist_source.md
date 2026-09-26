# Chat blocklist source

`chat_blocklist.json` is generated primarily from [Konsheng/Sensitive-lexicon](https://github.com/konsheng/Sensitive-lexicon) at commit `d967c30b053fa40b06c5a0dddf0be493f2dfae46`, licensed under MIT. The license is retained in `chat_blocklist_LICENSE.txt`. The 15 pre-existing local terms in `chat_blocklist_legacy.json` are preserved to avoid reducing previous coverage; they are not represented as upstream content.

Run `python tools/update_chat_blocklist.py` to regenerate the snapshot. Selected upstream files and their minimum normalized term lengths are in the script. Only terms containing Chinese characters and no longer than the 32-character live-chat limit are kept. This is a conservative keyword gate, not a comprehensive or context-aware moderation service; review false positives before changing the pinned version or selection rules.
