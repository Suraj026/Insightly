"""Generates answers from retrieved chunks using OpenRouter LLM."""
import re
from uuid import uuid4
from openai import AsyncOpenAI
from config import settings

class ResponseService:
    """Builds a citation-grounded prompt and calls the LLM."""
    SYSTEM_PROMPT = """
    You are a precise Q&A assistant. Answer the user's question based strictly on the provided context chunks.

    Rules:
    1. If the context doesn't contain enough information, say so clearly — do not speculate or use outside knowledge.
    2. Cite sources using [citation:N] where N is the chunk number.
    3. Use multiple citations when combining information from different chunks (e.g., [citation:1][citation:3]).
    4. Keep answers concise but thorough.
    5. Do not repeat or summarize the context chunks themselves — only answer the question asked.
    """
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=settings.openrouter_url,
            api_key=settings.openrouter_api_key
        )
        self.model = settings.openrouter_model

    def _build_prompt(self, query: str, chunks: list[dict]) -> str:
        """Constructs a prompt for the LLM based on the query and retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = f"[citation:{i+1}]"
            context_parts.append(f"{source} {chunk['chunk_text']}")
        context = "\n".join(context_parts)
        prompt = f"Context:\n{context}\n\n" \
                 f"Question: {query}\n\n" \
                 f"Answer (with citations):"
        
        return prompt
    
    def _parse_citations(self, answer: str, chunks: list[dict]) -> list[dict]:
        """Extracts cited sources from the answer and returns them with their content."""
        citations = []
        seen = set()

        for match in re.finditer(r"\[citation:(\d+)\]", answer):
            idx = int(match.group(1))
            if idx < len(chunks) and idx not in seen:
                seen.add(idx)
                chunk = chunks[idx]
                citations.append({
                    "chunk_text": chunk["chunk_text"],
                    "score": chunk.get("score", 0.0),
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                })

        return citations
    
    async def answer(self, query: str, chunks: list[dict]) -> dict:
        """Generates an answer to the query based on the provided chunks."""
        prompt = self._build_prompt(query, chunks)

        response = await self.client.chat.completions.create(
            model = self.model,
            messages = [
                {
                    "role": "system", 
                    "content": self.SYSTEM_PROMPT
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature = 0.3
        )

        answer_text = response.choices[0].message.content
        citations = self._parse_citations(answer_text, chunks)

        return {
            "trace_id": str(uuid4()),
            "answer": answer_text,
            "citations": citations,
        }