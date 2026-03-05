"""
Task Endpoints — /tasks/*

GET /tasks            List recent background tasks (test runs, scans, backfills)
GET /tasks/{task_id}  Poll a specific task for its result
"""

from fastapi import APIRouter, HTTPException, Query

from src.admin.task_manager import get_task_manager

tasks_router = APIRouter()


@tasks_router.get("")
async def list_tasks(limit: int = Query(default=20, ge=1, le=100)):
    """List recent background tasks, newest first."""
    return {"tasks": get_task_manager().list_tasks(limit=limit)}


@tasks_router.get("/{task_id}")
async def get_task(task_id: str):
    """Get current status and result for a specific task."""
    status = get_task_manager().get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail={"error": "task_not_found", "task_id": task_id})
    return status
