"""
AI Agent Router
Extracted from main.py - Handles AI assistant queries
"""
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["ai-agent"])

# Import agent
AI_AGENT_AVAILABLE = False
try:
    from ..agents.local_expert import agent
    AI_AGENT_AVAILABLE = True
    logger.info("AI Agent imported successfully")
except ImportError as e:
    logger.warning(f"AI Agent import failed: {e}")
    
    # Fallback mock agent
    class MockLocalExpertAgent:
        async def process_query(self, query: str):
            return {
                "query": query, 
                "answer": f"I'm your real estate assistant. Asked: {query}",
                "confidence": 0.85,
                "success": True
            }
    
    agent = MockLocalExpertAgent()


@router.post("/query")
async def query_agent(query_req: Dict[str, Any]):
    """
    Query AI agent for real estate analysis
    
    Request body:
        {
            "query": "Calculate ROI for $300,000 property with $2,000 rent",
            "context": {}  # optional
        }
    
    Returns:
        {
            "query": str,
            "response": dict,
            "timestamp": str,
            "confidence": float,
            "success": bool
        }
    """
    try:
        query = query_req.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        logger.info(f"AI Agent query: {query[:100]}")
        
        result = await agent.process_query(query)
        
        return {
            "query": query,
            "response": result,
            "timestamp": datetime.now().isoformat(),
            "confidence": result.get("confidence", 0.8),
            "success": result.get("success", True)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))