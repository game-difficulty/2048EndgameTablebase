import unittest
from unittest.mock import patch

from backend.trainer_default_lookup import TrainerDefaultLookup


class TrainerDefaultLookupTests(unittest.TestCase):
    def test_switch_cooldown_is_per_full_pattern(self):
        policy = TrainerDefaultLookup()
        with patch('backend.trainer_default_lookup.time.monotonic', return_value=100):
            ticket = policy.issue('free12_2048', 123, switched=True)
            self.assertEqual(policy.consume(ticket, 'free12_2048', 123), (True, True))
            ticket = policy.issue('free12_2048', 456, switched=True)
            self.assertEqual(policy.consume(ticket, 'free12_2048', 456), (True, False))
            ticket = policy.issue('free12_1024', 456, switched=True)
            self.assertEqual(policy.consume(ticket, 'free12_1024', 456), (True, True))
        with patch('backend.trainer_default_lookup.time.monotonic', return_value=700):
            ticket = policy.issue('free12_2048', 123, switched=True)
            self.assertEqual(policy.consume(ticket, 'free12_2048', 123), (True, True))

    def test_manual_default_does_not_use_free_allowance(self):
        policy = TrainerDefaultLookup()
        ticket = policy.issue('L3_256', 123, switched=False)
        self.assertEqual(policy.consume(ticket, 'L3_256', 123), (True, False))
        ticket = policy.issue('L3_256', 123, switched=True)
        self.assertEqual(policy.consume(ticket, 'L3_256', 123), (True, True))

    def test_ticket_is_bound_to_board_pattern_and_page_and_single_use(self):
        policy = TrainerDefaultLookup()
        ticket = policy.issue('L3_256', 123, switched=True)
        for token, pattern, board in [('', 'L3_256', 123), (ticket, 'L3_512', 123), (ticket, 'L3_256', 124)]:
            self.assertEqual(policy.consume(token, pattern, board), (False, False))
        self.assertEqual(TrainerDefaultLookup().consume(ticket, 'L3_256', 123), (False, False))
        self.assertEqual(policy.consume(ticket, 'L3_256', 123), (True, True))
        self.assertEqual(policy.consume(ticket, 'L3_256', 123), (False, False))

    def test_expired_and_replaced_tickets_are_rejected(self):
        policy = TrainerDefaultLookup()
        with patch('backend.trainer_default_lookup.time.monotonic', return_value=100):
            old = policy.issue('L3_256', 123, switched=True)
            ticket = policy.issue('L3_256', 123, switched=True)
            self.assertEqual(policy.consume(old, 'L3_256', 123), (False, False))
        with patch('backend.trainer_default_lookup.time.monotonic', return_value=221):
            self.assertEqual(policy.consume(ticket, 'L3_256', 123), (False, False))
