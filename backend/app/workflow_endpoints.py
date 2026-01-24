"""
FIXED: Async workflow endpoints with proper async/await
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import httpx
from datetime import datetime
import asyncio

router = APIRouter()

@router.post("/trigger-analysis")
async def trigger_analysis_workflow(
    address: str,
    radius_m: int = 1000,
    email: Optional[str] = None
):
    """
    FIXED: Trigger neighborhood analysis workflow with proper async
    """
    try:
        # Use httpx for async HTTP requests
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Start analysis
            analysis_response = await client.post(
                "http://localhost:8000/api/neighborhood/analyze",
                json={
                    "address": address,
                    "radius_m": radius_m,
                    "generate_map": True
                }
            )
            
            if analysis_response.status_code != 202:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Analysis failed to start: {analysis_response.text}"
                )
            
            analysis_data = analysis_response.json()
            analysis_id = analysis_data["analysis_id"]
            task_id = analysis_data["task_id"]
            
            # Poll for completion with proper async
            max_wait = 60  # seconds
            wait_interval = 2  # seconds
            max_attempts = max_wait // wait_interval
            
            for attempt in range(max_attempts):
                # ✅ FIXED: Use asyncio.sleep instead of time.sleep
                await asyncio.sleep(wait_interval)
                
                # Check status
                status_response = await client.get(
                    f"http://localhost:8000/api/tasks/{task_id}"
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status")
                    
                    if status == "completed":
                        # Get full results
                        result_response = await client.get(
                            f"http://localhost:8000/api/neighborhood/{analysis_id}"
                        )
                        
                        if result_response.status_code == 200:
                            result_data = result_response.json()
                            
                            # Send email if requested
                            if email:
                                await send_completion_email(
                                    email=email,
                                    address=address,
                                    walk_score=result_data.get('walk_score'),
                                    total_amenities=result_data.get('total_amenities', 0)
                                )
                            
                            return {
                                "workflow_id": f"workflow_{int(datetime.now().timestamp())}",
                                "status": "completed",
                                "analysis_id": analysis_id,
                                "task_id": task_id,
                                "result": {
                                    "walk_score": result_data.get('walk_score'),
                                    "total_amenities": result_data.get('total_amenities', 0),
                                    "address": address
                                },
                                "email_sent": bool(email),
                                "completed_at": datetime.now().isoformat()
                            }
                    
                    elif status == "failed":
                        error_msg = status_data.get('error', 'Unknown error')
                        
                        if email:
                            await send_failure_email(email, address, error_msg)
                        
                        raise HTTPException(
                            status_code=500,
                            detail=f"Analysis failed: {error_msg}"
                        )
                    
                    # Still processing, continue polling
            
            # Timeout reached
            if email:
                await send_timeout_email(email, address, task_id)
            
            return {
                "workflow_id": f"workflow_{int(datetime.now().timestamp())}",
                "status": "timeout",
                "analysis_id": analysis_id,
                "task_id": task_id,
                "message": "Analysis is taking longer than expected. Check task status.",
                "poll_url": f"/api/tasks/{task_id}",
                "email_sent": bool(email)
            }
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Backend service unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get workflow status by ID"""
    # In production, store workflow status in database
    return {
        "workflow_id": workflow_id,
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "message": "Workflow status tracking not yet implemented"
    }


@router.post("/webhook/analysis")
async def n8n_webhook_handler(payload: Dict[str, Any]):
    """
    FIXED: n8n webhook handler with proper validation
    """
    try:
        address = payload.get("address")
        if not address:
            raise HTTPException(status_code=400, detail="Address is required")
        
        radius_m = payload.get("radius_m", 1000)
        email = payload.get("email")
        
        # Validate inputs
        if not isinstance(radius_m, int) or radius_m < 100 or radius_m > 5000:
            raise HTTPException(
                status_code=400,
                detail="radius_m must be between 100 and 5000"
            )
        
        # Trigger workflow
        response = await trigger_analysis_workflow(
            address=address,
            radius_m=radius_m,
            email=email
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== EMAIL FUNCTIONS ====================

async def send_completion_email(
    email: str,
    address: str,
    walk_score: float,
    total_amenities: int
):
    """
    Send completion email
    In production, integrate with actual email service (SendGrid, AWS SES, etc.)
    """
    print(f"""
    ========================================
    EMAIL NOTIFICATION
    ========================================
    To: {email}
    Subject: GeoInsight Analysis Complete
    
    Your neighborhood analysis for {address} is complete!
    
    Results:
    - Walk Score: {walk_score:.1f}/100
    - Total Amenities: {total_amenities}
    
    View full results at: http://localhost:8501
    ========================================
    """)
    
    # TODO: Integrate actual email service
    # Example with SendGrid:
    # import sendgrid
    # from sendgrid.helpers.mail import Mail
    # sg = sendgrid.SendGridAPIClient(api_key=os.getenv('SENDGRID_API_KEY'))
    # message = Mail(...)
    # await sg.send(message)


async def send_failure_email(email: str, address: str, error: str):
    """Send failure notification"""
    print(f"""
    ========================================
    EMAIL NOTIFICATION (FAILURE)
    ========================================
    To: {email}
    Subject: GeoInsight Analysis Failed
    
    Your analysis for {address} failed.
    Error: {error}
    
    Please try again or contact support.
    ========================================
    """)


async def send_timeout_email(email: str, address: str, task_id: str):
    """Send timeout notification"""
    print(f"""
    ========================================
    EMAIL NOTIFICATION (TIMEOUT)
    ========================================
    To: {email}
    Subject: GeoInsight Analysis In Progress
    
    Your analysis for {address} is taking longer than expected.
    
    Track progress: http://localhost:8501
    Task ID: {task_id}
    ========================================
    """)


# ==================== BATCH WORKFLOW ====================

@router.post("/batch-analysis")
async def batch_analysis_workflow(
    addresses: list[str],
    radius_m: int = 1000,
    email: Optional[str] = None
):
    """
    NEW: Batch process multiple addresses
    """
    if len(addresses) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 addresses per batch"
        )
    
    results = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Start all analyses concurrently
        tasks = []
        for address in addresses:
            task = client.post(
                "http://localhost:8000/api/neighborhood/analyze",
                json={
                    "address": address,
                    "radius_m": radius_m,
                    "generate_map": False  # Skip maps for batch
                }
            )
            tasks.append(task)
        
        # Wait for all to start
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect task IDs
        for idx, response in enumerate(responses):
            if isinstance(response, Exception):
                results.append({
                    "address": addresses[idx],
                    "status": "failed",
                    "error": str(response)
                })
            elif response.status_code == 202:
                data = response.json()
                results.append({
                    "address": addresses[idx],
                    "status": "queued",
                    "task_id": data["task_id"],
                    "analysis_id": data["analysis_id"]
                })
            else:
                results.append({
                    "address": addresses[idx],
                    "status": "failed",
                    "error": f"HTTP {response.status_code}"
                })
    
    return {
        "batch_id": f"batch_{int(datetime.now().timestamp())}",
        "total": len(addresses),
        "results": results,
        "email_notification": bool(email)
    }