
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import requests
import json
from datetime import datetime

router = APIRouter(prefix="/api/workflow", tags=["workflow"])

@router.post("/trigger-analysis")
async def trigger_analysis_workflow(
    address: str,
    radius_m: int = 1000,
    email: str = None
):
    try:
        analysis_response = requests.post(
            "http://localhost:8000/api/neighborhood/analyze",
            json={
                "address": address,
                "radius_m": radius_m,
                "generate_map": True
            }
        )
        
        if analysis_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Analysis failed to start")
        
        analysis_data = analysis_response.json()
        analysis_id = analysis_data["analysis_id"]
        
        import time
        max_wait = 60  
        wait_interval = 2
        
        for _ in range(max_wait // wait_interval):
            
            status_response = requests.get(
                f"http://localhost:8000/api/neighborhood/{analysis_id}"
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data.get("status") == "completed":
                
                    if email:
                    
                        print(f"Would send email to {email}")
                        print(f" Analysis complete for {address}")
                        print(f" Walk Score: {status_data.get('walk_score')}")
                    
                    return {
                        "workflow_id": f"workflow_{datetime.now().timestamp()}",
                        "status": "completed",
                        "analysis_id": analysis_id,
                        "email_sent": bool(email)
                    }
            
            time.sleep(wait_interval)
        

        if email:
            print(f" Would send timeout email to {email}")
        
        return {
            "workflow_id": f"workflow_{datetime.now().timestamp()}",
            "status": "timeout",
            "analysis_id": analysis_id,
            "email_sent": bool(email)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):

    return {
        "workflow_id": workflow_id,
        "status": "completed", 
        "timestamp": datetime.now().isoformat()
    }

@router.post("/webhook/analysis")
async def n8n_webhook_handler(payload: Dict[str, Any]):

    try:
        address = payload.get("address")
        if not address:
            raise HTTPException(status_code=400, detail="Address is required")
  
        response = await trigger_analysis_workflow(
            address=address,
            radius_m=payload.get("radius_m", 1000),
            email=payload.get("email")
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))