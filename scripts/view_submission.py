#!/usr/bin/env python3
"""View submission details and code from the database.

Usage:
    # List all submissions
    uv run scripts/view_submission.py

    # View specific submission
    uv run scripts/view_submission.py commit_9303_1

    # Save code to file
    uv run scripts/view_submission.py commit_9303_1 --save

    # Filter by spec_version
    uv run scripts/view_submission.py --version 2
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def list_submissions(db_path: str, spec_version: int | None = None):
    """List all submissions in the database."""
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        query = """
            SELECT submission_id, miner_uid, spec_version, status, final_score, created_at,
                   LENGTH(code_content) as code_len
            FROM submissions
        """
        params = ()
        if spec_version is not None:
            query += " WHERE spec_version = ?"
            params = (spec_version,)
        query += " ORDER BY created_at DESC"

        cur.execute(query, params)
        rows = cur.fetchall()

        if not rows:
            print("No submissions found.")
            return

        print("=" * 90)
        print(
            f"{'SUBMISSION':<25} {'UID':<5} {'VER':<4} {'STATUS':<10} {'MFU':<10} {'CODE':<10} {'SUBMITTED'}"
        )
        print("=" * 90)

        for r in rows:
            mfu = f"{r['final_score']:.2f}%" if r["final_score"] else "N/A"
            code = f"{r['code_len']}b" if r["code_len"] else "N/A"
            created = r["created_at"][:19] if r["created_at"] else "N/A"
            print(
                f"{r['submission_id']:<25} {r['miner_uid']:<5} {r['spec_version']:<4} "
                f"{r['status']:<10} {mfu:<10} {code:<10} {created}"
            )
    finally:
        conn.close()


def view_submission(db_path: str, submission_id: str, save: bool = False):
    """View a specific submission's details and code."""
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT submission_id, miner_uid, miner_hotkey, spec_version, status, final_score,
                   code_hash, created_at, code_content
            FROM submissions
            WHERE submission_id = ?
        """,
            (submission_id,),
        )
        row = cur.fetchone()

        if not row:
            print(f"Submission '{submission_id}' not found.")
            print("\nAvailable submissions:")
            list_submissions(db_path)
            return

        print("=" * 70)
        print(f"SUBMISSION: {row['submission_id']}")
        print("=" * 70)
        print(f"UID:        {row['miner_uid']}")
        print(f"Hotkey:     {row['miner_hotkey']}")
        print(f"Version:    {row['spec_version']}")
        print(f"Status:     {row['status']}")
        if row["final_score"]:
            print(f"MFU:        {row['final_score']:.2f}%")
        print(f"Code URL:   {row['code_hash']}")
        print(f"Submitted:  {row['created_at']}")

        # Get evaluations for this submission
        cur.execute(
            """
            SELECT evaluation_id, mfu, tokens_per_second, total_tokens, 
                   wall_time_seconds, success, error, created_at
            FROM evaluations
            WHERE submission_id = ?
            ORDER BY created_at
        """,
            (submission_id,),
        )
        evals = cur.fetchall()

        if evals:
            print()
            print("-" * 70)
            print("EVALUATIONS:")
            print("-" * 70)
            for i, e in enumerate(evals, 1):
                status = "PASS" if e["success"] else "FAIL"
                print(
                    f"  #{i}: MFU={e['mfu']:.2f}% TPS={e['tokens_per_second']:.0f} "
                    f"tokens={e['total_tokens']} time={e['wall_time_seconds']:.2f}s [{status}]"
                )
                if e["error"]:
                    print(f"      Error: {e['error'][:60]}...")

        print()

        if row["code_content"]:
            if save:
                # Save to file
                filename = f"{submission_id}_train.py"
                with open(filename, "w") as f:
                    f.write(row["code_content"])
                print(f"Code saved to: {filename}")
            else:
                print("=" * 70)
                print("CODE CONTENT:")
                print("=" * 70)
                print(row["code_content"])
                print()
                print("(Use --save to save code to a file)")
        else:
            print("(Code not yet stored - evaluation in progress)")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="View submission details and code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all submissions
  uv run scripts/view_submission.py

  # List submissions for spec_version 2
  uv run scripts/view_submission.py --version 2

  # View specific submission
  uv run scripts/view_submission.py commit_9303_1

  # Save code to file
  uv run scripts/view_submission.py commit_9303_1 --save
        """,
    )
    parser.add_argument("submission_id", nargs="?", help="Submission ID to view")
    parser.add_argument("--save", action="store_true", help="Save code to file")
    parser.add_argument("--db", default="crusades.db", help="Database path")
    parser.add_argument(
        "--version", "-v", type=int, help="Filter by spec_version (competition version)"
    )

    args = parser.parse_args()

    # Find database
    db_path = args.db
    if not Path(db_path).exists():
        # Try from project root
        project_root = Path(__file__).parent.parent
        db_path = project_root / "crusades.db"
        if not db_path.exists():
            print(f"Database not found: {args.db}")
            sys.exit(1)
        db_path = str(db_path)

    if args.submission_id:
        view_submission(db_path, args.submission_id, args.save)
    else:
        list_submissions(db_path, args.version)


if __name__ == "__main__":
    main()
