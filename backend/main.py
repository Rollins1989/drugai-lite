"""DrugAI Lite application entry point."""
import time, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from api.routes import router
from config import APP_VERSION, STATIC_DIR

app=FastAPI(title="DrugAI Lite",version=APP_VERSION,description="Open-source computational drug-discovery workspace for molecular profiling, target activity prediction and virtual screening.")
app.include_router(router)

@app.middleware("http")
async def request_metadata(request:Request,call_next):
    request_id=request.headers.get("X-Request-ID",uuid.uuid4().hex[:16])
    start=time.perf_counter()
    try:
        response=await call_next(request)
    except Exception:
        response=JSONResponse(status_code=500,content={"detail":"Internal server error","request_id":request_id})
    response.headers["X-Request-ID"]=request_id
    response.headers["X-Process-Time-Ms"]=str(round((time.perf_counter()-start)*1000,2))
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["X-Frame-Options"]="DENY"
    response.headers["Referrer-Policy"]="strict-origin-when-cross-origin"
    return response

app.mount("/",StaticFiles(directory=str(STATIC_DIR),html=True),name="static")
