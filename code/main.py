import os
import time
import pandas as pd
from retriever import DocumentRetriever
from agent import SupportAgent


def process_tickets(input_csv_path, output_csv_path):
    # ─── Step 1: Load & Validate Input ───────────────────────────────────────
    print(f"Loading support tickets from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    # Pre-flight: warn if row count is wrong (known bug: embedded newlines in
    # Issue field cause pandas to silently drop rows — see GitHub Issue #28).
    # Run fix_csv.py to repair the CSV if this warning appears.
    EXPECTED_ROWS = 30
    if len(df) < EXPECTED_ROWS:
        print(f"[WARNING] CSV parsed as {len(df)} rows, expected {EXPECTED_ROWS}.")
        print(f"[WARNING] The CSV may be malformed. Run fix_csv.py to repair it.")
    else:
        print(f"[OK] {len(df)} ticket(s) loaded.")

    # Initialize RAG components once, reuse across all tickets
    retriever = DocumentRetriever(data_dir="../data")
    agent = SupportAgent()

    results = []
    total = len(df)

    # ─── Step 7: Sequential Execution (one ticket at a time) ─────────────────
    for index, row in df.iterrows():
        ticket_num = index + 1
        issue   = str(row.get('Issue', ''))
        subject = str(row.get('Subject', ''))
        company = str(row.get('Company', 'None'))

        print(f"\n[{ticket_num}/{total}] Processing ticket (Company: {company})...")

        # ─── Step 2: Document Retrieval (top 3 chunks, filtered by company) ──
        query = f"{subject} {issue}"
        docs  = retriever.retrieve(query, company=company, top_k=3)

        # ─── Steps 3-5: Response Generation + Fallback + Safety Verification ─
        prediction = agent.process_ticket(issue, subject, company, docs)

        # ─── Step 6: Collect Structured Output ───────────────────────────────
        results.append({
            'status':        prediction.get('status',        'escalated'),
            'product_area':  prediction.get('product_area',  'unknown'),
            'response':      prediction.get('response',      'Unable to process.'),
            'justification': prediction.get('justification', 'Error'),
            'request_type':  prediction.get('request_type',  'product_issue'),
        })

        print(f"  → Status: {results[-1]['status']} | Type: {results[-1]['request_type']}")

    # ─── Step 6: Write Output CSV ─────────────────────────────────────────────
    print(f"\nSaving predictions to {output_csv_path}...")
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:  # Guard against empty string when path has no directory prefix
        os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    replied   = sum(1 for r in results if r['status'] == 'replied')
    escalated = sum(1 for r in results if r['status'] == 'escalated')
    print(f"Done! {total} ticket(s) processed: {replied} replied, {escalated} escalated.")


if __name__ == "__main__":
    input_file  = "../support_tickets/support_tickets.csv"
    output_file = "../support_tickets/output.csv"
    process_tickets(input_file, output_file)
