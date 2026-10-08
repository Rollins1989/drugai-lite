"""DrugAI Lite application entry point."""
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest\n\ndef unique_operation_id(route):\n    methods=sorted(getattr(route,"methods",[]) or [])\n    method="_".join(m.lower() for m in methods)\n    return f"{method}_{route.path_format.strip("/").replace("/", "_").replace("{", "").replace("}", "")}"
from api.routes import router
from config import APP_VERSION, STATIC_DIR

REQUEST_COUNT=Counter("drugai_http_requests_total","HTTP requests",["method","path","status"])
REQUEST_LATENCY=Histogram("drugai_http_request_duration_seconds","HTTP request latency",["method","path"])
app=FastAPI(title="DrugAI Lite",version=APP_VERSION,generate_unique_id_function=unique_operation_id,description="Open-source computational drug-discovery workspace for molecular profiling, target activity prediction and virtual screening.")
app.include_router(router,prefix="/api/v1")
# Backward-compatible legacy surface. New integrations should use /api/v1.
app.include_router(router,prefix="/api")

@app.middleware("http")
async def request_metadata(request:Request,call_next):
    request_id=request.headers.get("X-Request-ID",uuid.uuid4().hex[:16])
    start=time.perf_counter()
    try:
        response=await call_next(request)
    except Exception:
        response=JSONResponse(status_code=500,content={"detail":"Internal server error","request_id":request_id})
    response.headers["X-Request-ID"]=request_id
    elapsed=time.perf_counter()-start
    response.headers["X-Process-Time-Ms"]=str(round(elapsed*1000,2))
    metric_path=getattr(request.scope.get("route"),"path",request.url.path)
    REQUEST_COUNT.labels(request.method,metric_path,str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method,metric_path).observe(elapsed)
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["X-Frame-Options"]="DENY"
    response.headers["Referrer-Policy"]="strict-origin-when-cross-origin"
    return response

app.mount("/",StaticFiles(directory=str(STATIC_DIR),html=True),name="static")


@app.get("/api/v1/metrics",include_in_schema=False)
def metrics():
    return Response(content=generate_latest(),media_type=CONTENT_TYPE_LATEST)
