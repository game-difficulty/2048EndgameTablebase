"""Compatibility import for the reusable room activity."""
import sys
from backend.room_activities import red_envelopes
sys.modules[__name__] = red_envelopes
