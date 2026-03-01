"""
Optional LLM API Endpoint for Node.js Backend
==============================================

This file adds an OPTIONAL endpoint to the vision engine
that Node.js backend can call to get LLM analysis.

IMPORTANT:
- Python vision engine NEVER calls this automatically
- Only responds when Node.js explicitly requests analysis
- Keeps Python's main detection loop fast and clean

Usage:
    Add to streaming_server_integrated.py:

    from llm_api_endpoint import router as llm_router
    app.include_router(llm_router)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Import LLM analyzer (only loaded if this endpoint is used)
try:
    from brain.llm_analyzer import LLMAnalyzer
    llm_analyzer = LLMAnalyzer()
    LLM_AVAILABLE = True
    logger.info("✓ LLM analyzer loaded for API endpoint")
except ImportError:
    LLM_AVAILABLE = False
    llm_analyzer = None
    logger.warning("⚠ LLM analyzer not available")


class AnalysisRequest(BaseModel):
    """Request body for LLM analysis"""
    track_id: int
    camera_id: str
    risk_score: int
    features: Dict


class AnalysisResponse(BaseModel):
    """Response with LLM reasoning"""
    risk_level: str
    confidence: float
    reasoning: str
    alert_message: str
    recommended_action: str


@router.post("/api/analyze-alert", response_model=AnalysisResponse)
async def analyze_alert(request: AnalysisRequest):
    """
    LLM Analysis Endpoint (CALLED BY NODE.JS BACKEND ONLY)

    This endpoint is NEVER called by Python's detection loop.
    Only Node.js backend calls this when it needs LLM reasoning.

    Flow:
    1. Python detects person with risk_score = 68
    2. Python sends detection to Node.js
    3. Node.js sees 50 <= 68 < 85 → needs LLM
    4. Node.js calls this endpoint
    5. Python calls Ollama and returns reasoning
    6. Node.js saves alert with LLM reasoning

    Args:
        request: Detection data from Node.js

    Returns:
        LLM analysis with reasoning

    Raises:
        503: If LLM not available
        500: If LLM analysis fails
    """
    if not LLM_AVAILABLE or llm_analyzer is None:
        logger.error("LLM analysis requested but LLM not available")
        raise HTTPException(
            status_code=503,
            detail="LLM analyzer not available. Install Ollama and run 'ollama pull llama3.1:8b'"
        )

    try:
        logger.info(f"📡 LLM analysis requested for track {request.track_id}")

        # Call LLM analyzer
        result = llm_analyzer.analyze(
            features=request.features,
            risk_score=request.risk_score,
            track_id=request.track_id,
            camera_id=request.camera_id
        )

        logger.info(f"✅ LLM analysis complete for track {request.track_id}")

        return AnalysisResponse(
            risk_level=result['risk_level'],
            confidence=result['confidence'],
            reasoning=result['reasoning'],
            alert_message=result['alert_message'],
            recommended_action=result['recommended_action']
        )

    except Exception as e:
        logger.error(f"❌ LLM analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM analysis failed: {str(e)}"
        )


@router.get("/api/llm-status")
async def llm_status():
    """
    Check if LLM is available and working

    Returns:
        Status of LLM analyzer
    """
    if not LLM_AVAILABLE:
        return {
            "available": False,
            "message": "LLM analyzer not loaded. Install brain modules."
        }

    if llm_analyzer is None:
        return {
            "available": False,
            "message": "LLM analyzer failed to initialize"
        }

    # Test Ollama connection
    try:
        is_connected = llm_analyzer._test_connection()
        if is_connected:
            return {
                "available": True,
                "message": "LLM analyzer ready",
                "model": llm_analyzer.model,
                "base_url": llm_analyzer.base_url
            }
        else:
            return {
                "available": False,
                "message": "Cannot connect to Ollama. Run 'ollama serve'"
            }
    except Exception as e:
        return {
            "available": False,
            "message": f"Error testing Ollama: {str(e)}"
        }


# ============================================================================
# INTEGRATION INSTRUCTIONS
# ============================================================================
"""
To enable this endpoint in streaming_server_integrated.py:

1. Add import at top:
   from llm_api_endpoint import router as llm_router

2. Include router after creating app:
   app.include_router(llm_router)

3. Start server:
   python streaming_server_integrated.py

4. Test endpoint:
   curl http://localhost:5000/api/llm-status

   curl -X POST http://localhost:5000/api/analyze-alert \
     -H "Content-Type: application/json" \
     -d '{
       "track_id": 123,
       "camera_id": "camera_1",
       "risk_score": 68,
       "features": {
         "min_dist_to_edge": 75.0,
         "dwell_time_near_edge": 8.2
       }
     }'

IMPORTANT:
- Python detection loop NEVER calls this endpoint
- Only Node.js backend calls it (when needed)
- Keeps Python fast for real-time video processing
"""
