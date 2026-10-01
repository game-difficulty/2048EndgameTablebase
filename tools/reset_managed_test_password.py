from __future__ import annotations

import argparse
from getpass import getpass

from backend.auth.managed_test_accounts import reset_managed_password


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset an admin-managed test account password")
    parser.add_argument("user_id", type=int)
    parser.add_argument("--operator-id", required=True, type=int)
    args = parser.parse_args()
    password = getpass("New password: ")
    if password != getpass("Confirm password: "):
        parser.error("passwords do not match")
    reset_managed_password(
        user_id=args.user_id, operator_id=args.operator_id, new_password=password
    )
    print("Password updated; all account sessions revoked.")


if __name__ == "__main__":
    main()
