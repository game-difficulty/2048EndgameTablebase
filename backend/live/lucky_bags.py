"""Compatibility import for the reusable room activity."""
import sys
from backend.room_activities import lucky_bags
sys.modules[__name__] = lucky_bags
