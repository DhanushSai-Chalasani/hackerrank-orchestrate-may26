import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SupportAgent:
    def __init__(self):
        # Load all API keys from .env, separated by comma
        keys_str = os.environ.get("GROQ_API_KEYS", "")
        self.api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not self.api_keys:
            raise ValueError("No GROQ_API_KEYS found in .env")
        
        self.current_key_idx = 0
        self._init_client()
        
        # Upgrading back to the massive 70B model since we have multi-key failover!
        self.model = 'llama-3.3-70b-versatile'
        # Retries equal to 2x the number of keys we have
        self.max_retries = max(3, len(self.api_keys) * 2)

    def _init_client(self):
        self.client = OpenAI(
            api_key=self.api_keys[self.current_key_idx],
            base_url="https://api.groq.com/openai/v1"
        )
        
    def _switch_key(self):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        print(f"  [Key Switched] Rotating to API key #{self.current_key_idx + 1}", flush=True)
        self._init_client()

    def _call_with_retry(self, system_prompt, user_prompt):
        """Call the Groq API with failover retry on rate limit errors."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e)
                is_retryable = "429" in error_str or "503" in error_str or "rate_limit" in error_str.lower()
                if is_retryable and attempt < self.max_retries - 1:
                    print(f"  [Rate Limited] Switching API key to bypass limit...", flush=True)
                    self._switch_key()
                    time.sleep(1) # Small delay before retrying with new key
                else:
                    raise e
        raise Exception("Max retries exceeded.")

    def _verify_safety(self, draft_response, context):
        """Guardrail: Ensure the response is grounded and safe."""
        if not draft_response:
            return False
            
        verify_sys = "You are a strict compliance auditor. Verify if the proposed response makes ANY claims that are NOT explicitly supported by the provided context. If it hallucinates or provides outside info, output 'UNSAFE'. Otherwise 'SAFE'."
        verify_user = f"Context:\n{context}\n\nProposed Response:\n{draft_response}\n\nOutput ONLY 'SAFE' or 'UNSAFE'."
        
        for attempt in range(self.max_retries):
            try:
                res = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": verify_sys},
                        {"role": "user", "content": verify_user}
                    ],
                    temperature=0.0
                )
                decision = res.choices[0].message.content.strip().upper()
                return "UNSAFE" not in decision
            except Exception as e:
                error_str = str(e)
                is_retryable = "429" in error_str or "503" in error_str or "rate_limit" in error_str.lower()
                if is_retryable and attempt < self.max_retries - 1:
                    print(f"  [Guardrail Rate Limited] Switching API key...", flush=True)
                    self._switch_key()
                    time.sleep(1)
                else:
                    print(f"  [Warning] Guardrail check failed: {e}")
                    return False # Default to unsafe if we can't verify

    def process_ticket(self, issue, subject, company, retrieved_docs):
        context = ""
        for i, doc in enumerate(retrieved_docs):
            context += f"\n--- Document {i+1} from {doc['company']} ---\n{doc['content']}\n"

        system_prompt = """You are a highly accurate Support Triage Agent for HackerRank, Claude, and Visa.
Your goal is to classify support tickets and provide responses strictly based on the provided corpus.

CRITICAL RULES:
1. No outside knowledge: Your response MUST be grounded entirely in the provided document context.
2. Escalation: If the issue involves high-risk topics (fraud, billing, identity theft, sensitive account access, score disputes) OR if the answer is NOT found in the context, you MUST set status to "escalated". If you escalate, the response should briefly state that the issue has been escalated.
3. Allowed Output Values:
   - status: "replied" or "escalated"
   - request_type: "product_issue", "feature_request", "bug", or "invalid"
4. If the user is just saying hi or thanks, or asking completely unrelated things, set request_type to "invalid".
5. Chain of Thought: You MUST use the 'thought_process' key to explain your reasoning step-by-step before finalizing the other fields. Analyze if the issue is high-risk, and check if the context truly contains the answer.

Output your result exactly as a JSON object with these keys:
{
  "thought_process": "...",
  "status": "...",
  "product_area": "...",
  "response": "...",
  "justification": "...",
  "request_type": "..."
}
Do not include any text outside the JSON object.
"""

        user_prompt = f"""
Support Ticket details:
Subject: {subject}
Issue: {issue}
Company hint: {company}

Here is the retrieved context from the official support documents:
{context}

Based on the rules and context, analyze this ticket and provide the JSON output.
"""

        try:
            output_text = self._call_with_retry(system_prompt, user_prompt)
            result = json.loads(output_text)
            
            # Hallucination Guardrail Verification
            if result.get("status") == "replied":
                is_safe = self._verify_safety(result.get("response"), context)
                if not is_safe:
                    print("  [GUARDRAIL TRIGGERED] Hallucination or unsafe content detected. Escalating.", flush=True)
                    result["status"] = "escalated"
                    result["response"] = "The issue has been escalated to a human agent."
                    result["justification"] = "Guardrail detected potential hallucination or unsafe information."
                    
            return result

        except Exception as e:
            print(f"  [FAILED] Could not process ticket after {self.max_retries} retries: {e}")
            return {
                "status": "escalated",
                "product_area": "unknown",
                "response": "Unable to process request at this time. Escalating to human support.",
                "justification": f"LLM unavailable after retries: {str(e)[:200]}",
                "request_type": "product_issue"
            }

