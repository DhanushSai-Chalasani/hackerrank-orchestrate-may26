import os
import pandas as pd
from retriever import DocumentRetriever
from agent import SupportAgent
import time

def process_tickets(input_csv_path, output_csv_path):
    print(f"Loading support tickets from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    # Initialize the RAG system and Agent
    retriever = DocumentRetriever(data_dir="../data")
    agent = SupportAgent()
    
    results = []
    
    # Process each row
    for index, row in df.iterrows():
        issue = str(row.get('Issue', ''))
        subject = str(row.get('Subject', ''))
        company = str(row.get('Company', 'None'))
        
        print(f"Processing ticket {index+1}/{len(df)} (Company: {company})...", flush=True)
        
        # 1. Retrieve relevant context
        # Combine issue and subject for better retrieval
        query = f"{subject} {issue}"
        docs = retriever.retrieve(query, company=company, top_k=5)
        
        # 2. Call the LLM Agent
        prediction = agent.process_ticket(issue, subject, company, docs)
        
        # 3. Store results
        results.append({
            'status': prediction.get('status', 'escalated'),
            'product_area': prediction.get('product_area', 'unknown'),
            'response': prediction.get('response', 'Unable to process.'),
            'justification': prediction.get('justification', 'Error'),
            'request_type': prediction.get('request_type', 'product_issue')
        })
        
        # Adding a 3s delay to avoid hitting Groq free-tier rate limits (30 RPM for 8b-instant)
        time.sleep(3)
        
    print(f"Saving predictions to {output_csv_path}...")
    # The output format expects the 5 columns, but typically the problem statement 
    # asks for exactly these columns in the output CSV.
    output_df = pd.DataFrame(results)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    
    print("Done! Predictions saved successfully.")

if __name__ == "__main__":
    input_file = "../support_tickets/support_tickets.csv"
    output_file = "../support_tickets/output.csv"
    process_tickets(input_file, output_file)
