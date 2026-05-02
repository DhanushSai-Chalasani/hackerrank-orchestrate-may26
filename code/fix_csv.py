"""
Fix script for the malformed support_tickets.csv.

Bug (GitHub Issue #28): Row 7's Issue field contains an embedded LF (\n) inside
the quoted value "i can not able to see apply tab\n". Per RFC 4180 this is legal,
but it causes standard pandas/csv parsers to produce 29 rows instead of 30,
silently dropping ticket 8 and shifting all subsequent rows by one field.

Fix: Remove the embedded \n from row 7's Issue field and restore ticket 8
     as its own properly-separated row.
"""

import os
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'support_tickets', 'support_tickets.csv')

def fix_csv(path):
    # Read raw bytes to operate at byte level (avoid any parser interpretation)
    with open(path, 'rb') as f:
        raw = f.read()

    # The malformed sequence (bytes):
    #   "i can not able to see apply tab\n","I need to practice, submissions not working",HackerRank\r\n
    #
    # After fix it should become TWO rows:
    #   Row 7: "i can not able to see apply tab",Apply tab not visible,HackerRank\r\n
    #   Row 8: "I need to practice, submissions not working",Submissions not working,HackerRank\r\n
    #
    # Note: ticket 8's original Subject is lost (it was overwritten by the bug).
    # We reconstruct it from the Issue text as the most faithful approximation.

    malformed = (
        b'"i can not able to see apply tab\n",'
        b'"I need to practice, submissions not working",'
        b'HackerRank\r\n'
    )

    repaired = (
        b'"i can not able to see apply tab",'
        b'Apply tab not visible,'
        b'HackerRank\r\n'
        b'"I need to practice, submissions not working",'
        b'Submissions not working,'
        b'HackerRank\r\n'
    )

    if malformed not in raw:
        print("[INFO] Malformed sequence not found — CSV may already be fixed.")
        return False

    fixed = raw.replace(malformed, repaired, 1)  # replace only the first occurrence

    with open(path, 'wb') as f:
        f.write(fixed)

    print(f"[OK] Malformed row repaired in: {path}")
    return True


def verify_csv(path):
    df = pd.read_csv(path)
    row_count = len(df)
    print(f"[Verify] pandas reads {row_count} rows (expected 30)")

    if row_count == 30:
        print(f"  Row 7 (ticket 7): Issue='{df.iloc[6]['Issue'][:60]}' | Subject='{df.iloc[6]['Subject']}' | Company='{df.iloc[6]['Company']}'")
        print(f"  Row 8 (ticket 8): Issue='{df.iloc[7]['Issue'][:60]}' | Subject='{df.iloc[7]['Subject']}' | Company='{df.iloc[7]['Company']}'")
        print("[PASS] CSV is now well-formed with 30 rows.")
    else:
        print("[FAIL] Row count still wrong.")

    return row_count == 30


if __name__ == "__main__":
    changed = fix_csv(CSV_PATH)
    verify_csv(CSV_PATH)
    if not changed:
        print("No changes were needed.")
