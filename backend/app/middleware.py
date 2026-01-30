"""
Custom Middleware for Request/Response Processing
Handles request validation, logging, and security headers
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging
import time
from typing import Callable
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate all incoming requests
    - Check request size
    - Validate content-type
    - Sanitize headers
    """
    
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
    
    async def dispatch(self, request: Request, call_next: Callable) -> Callable:
        """Process request before it reaches the endpoint"""
        
        try:
            # 1. Generate request ID for tracking
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            
            # 2. Check content length
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.MAX_CONTENT_LENGTH:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request too large",
                        "max_size": f"{self.MAX_CONTENT_LENGTH / 1_000_000}MB",
                        "request_id": request_id
                    }
                )
            
            # 3. Validate content-type for POST/PUT/PATCH
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "")
                if not content_type or not any(ct in content_type for ct in 
                    ["application/json", "multipart/form-data", "application/x-www-form-urlencoded"]):
                    return JSONResponse(
                        status_code=415,
                        content={
                            "error": "Unsupported media type",
                            "received": content_type,
                            "supported": ["application/json", "multipart/form-data"],
                            "request_id": request_id
                        }
                    )
            
            # 4. Add security headers to response
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Request validation error: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": "Bad request", "detail": str(e)}
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all requests and responses
    Helps with debugging and monitoring
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Callable:
        """Log request details"""
        
        # Start timer
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        logger.info(
            f"📨 REQUEST [ID: {request_id}] {request.method} {request.url.path} | "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            
            # Log response
            status_indicator = "✅" if 200 <= response.status_code < 300 else \
                              "⚠️ " if 300 <= response.status_code < 400 else \
                              "❌"
            
            logger.info(
                f"{status_indicator} RESPONSE [ID: {request_id}] "
                f"Status: {response.status_code} | Time: {process_time:.3f}s"
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"❌ ERROR [ID: {request_id}] {request.method} {request.url.path} | "
                f"Error: {str(e)} | Time: {process_time:.3f}s"
            )
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Callable:
        """Add security headers"""
        
        response = await call_next(request)
        
        # Security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class RateLimitHeaderMiddleware(BaseHTTPMiddleware):
    """
    Add rate limit information to response headers
    Helps clients understand their rate limit status
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Callable:
        """Add rate limit headers to response"""
        
        response = await call_next(request)
        
        # These would be populated by slowapi rate limiter
        # This is a placeholder for proper implementation
        if hasattr(request.state, "rate_limit_info"):
            info = request.state.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(info.get("limit", 60))
            response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 60))
            response.headers["X-RateLimit-Reset"] = str(info.get("reset", 0))
        
        return response
