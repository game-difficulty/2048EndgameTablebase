import unittest
from backend.live.chat_length import chat_length, LIMIT


class ChatLengthTests(unittest.TestCase):
    def test_weighted_boundary_and_mixed_text(self):
        self.assertEqual(LIMIT, 80)
        for text in ('a' * 80, '中' * 40, '中' * 20 + 'a' * 40):
            self.assertEqual(chat_length(text), 80)
        self.assertEqual(chat_length('hello 中文!'), 11)

    def test_emoji_clusters_and_accents(self):
        for text in ('😀', '👍🏽', '👨‍👩‍👧‍👦', '🇨🇳', '❤️'):
            self.assertEqual(chat_length(text), 2)
        self.assertEqual(chat_length('e\u0301'), 1)
