import logging
import re
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from execution import (
    NaturalLanguageExecutor,
    start_session,
    resume_session,
    terminate_session,
)

app = FastAPI()
logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

ALLOWED_COMMANDS = {
    "load",
    "clean",
    "encode",
    "scale",
    "split",
    "build",
    "fit",
    "transform",
    "evaluate",
    "save",
    "train",
    "reset",
}

class CommandRequest(BaseModel):
    command: str
    session_id: str | None = None

class CommandResponse(BaseModel):
    session_id: str
    success: bool
    output: str
    error: str | None = None


def _sanitize_error(msg: str) -> str:
    return re.sub(r"/[^\s]+", "<path>", msg)


def _get_session(session_id: str | None):
    if session_id:
        session = resume_session(session_id)
        if session:
            return session
    return start_session()


@app.post("/execute", response_model=CommandResponse)
def execute(req: CommandRequest):
    start = time.time()
    cmd = req.command.strip()
    first = cmd.split(" ", 1)[0].lower()

    if cmd.lower() == "reset session":
        if req.session_id:
            terminate_session(req.session_id)
        session = start_session()
        logger.info("cmd='reset session' duration=%.3fs success=True", time.time() - start)
        return CommandResponse(session_id=session.id, success=True, output="Session reset.")

    if first not in ALLOWED_COMMANDS:
        logger.warning("cmd=%s not allowed", cmd)
        raise HTTPException(status_code=400, detail="Command not permitted.")

    session = _get_session(req.session_id)
    executor = NaturalLanguageExecutor()
    executor.context = session.context

    try:
        result = executor.execute(cmd)
        success = not result.startswith("✗")
        duration = time.time() - start
        logger.info(
            "cmd=%s duration=%.3fs success=%s",
            cmd,
            duration,
            success,
        )
        return CommandResponse(
            session_id=session.id,
            success=success,
            output=result if success else "",
            error=None if success else result,
        )
    except ValueError as exc:
        duration = time.time() - start
        msg = _sanitize_error(str(exc))
        logger.error("cmd=%s duration=%.3fs error=%s", cmd, duration, msg)
        return CommandResponse(
            session_id=session.id,
            success=False,
            output="",
            error=msg,
        )
    except Exception as exc:
        duration = time.time() - start
        msg = _sanitize_error(str(exc))
        logger.error("cmd=%s duration=%.3fs error=%s", cmd, duration, msg)
        raise HTTPException(status_code=500, detail=f"Execution failed: {msg}")


@app.post("/parse", response_model=CommandResponse)
def parse(req: CommandRequest):
    return execute(req)
