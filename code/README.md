# AI Support Triage Agent

This is a terminal-based AI agent designed for the HackerRank Orchestrate hackathon. It processes support tickets across three product ecosystems (HackerRank, Claude, and Visa), retrieves relevant grounding information from the provided support corpus using a local offline Sentence-Transformers model, and generates structured outputs (responses, routing, and justification) using an LLM.

## Requirements

The project uses Python. Ensure you have Python 3.9+ installed.

## Setup

1. **Install Dependencies**
   Navigate to the `code/` directory and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   The agent requires an LLM API key. Copy the `.env.example` file to `.env` and configure your API keys:
   ```bash
   cp .env.example .env
   ```
   *Note: Open `.env` and add your valid API key (e.g. `OPENAI_API_KEY` or equivalent).*

## Running the Agent

To process the support tickets, simply run:
```bash
python main.py
```

The agent will:
1. Load the support tickets from `../support_tickets/support_tickets.csv`.
2. Automatically build a local semantic index (saving to `.embeddings_cache/`) from the `../data/` corpus on the first run. Subsequent runs will use the cache.
3. Use semantic search to fetch the most relevant context chunks for each ticket.
4. Triage and process each ticket based on strict safety and resolution guidelines.
5. Save the final predictions to `../support_tickets/output.csv`.

## Project Structure
- `main.py`: Entry point for the application. Iterates through the CSV and orchestrates the pipeline.
- `agent.py`: Handles interactions with the LLM and enforces strict routing and escalation rules.
- `retriever.py`: Manages document chunking, caching, and offline embedding-based semantic retrieval using `sentence-transformers`.
- `fix_csv.py`: Helper script to repair malformed input CSV files if needed.
- `requirements.txt`: Python package dependencies.
