from fastapi import FastAPI
from .models import create_tables # type: ignore
from .routers import keys, security, health # type: ignore

app = FastAPI()

# Create tables
create_tables()

# Root route
@app.get("/")
def public():
    return {"message": "This is a public endpoint"}

# Include routers
app.include_router(keys.router)
app.include_router(security.router)
app.include_router(health.router)
