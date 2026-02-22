from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router

app = FastAPI(
    title="Dynamic Pricing System for Online Stores",
    description="ML + RL based real-time pricing optimization service.",
    version="1.0.0",
)

app.include_router(router)
app.mount('/static', StaticFiles(directory='app/static'), name='static')
templates = Jinja2Templates(directory='app/templates')


@app.get('/', response_class=HTMLResponse, tags=['dashboard'])
def dashboard(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})
