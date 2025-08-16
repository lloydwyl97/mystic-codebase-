from fastapi import FastAPI, APIRouter

# Create a simple test router
test_router = APIRouter()


@test_router.get("/test")
async def test():
    return {"message": "test"}


print(f"Test router has {len(test_router.routes)} routes")

# Create FastAPI app
app = FastAPI()

# Include test router
app.include_router(test_router, prefix="/api")
print(f"After including test router, app has {len(app.routes)} routes")

# Try to include ai_strategy_router
try:
    from ai_strategy_endpoints import router as ai_strategy_router

    print(f"AI strategy router has {len(ai_strategy_router.routes)} routes")

    app.include_router(ai_strategy_router, prefix="/api")
    print(f"After including AI strategy router, app has {len(app.routes)} routes")

    # List all routes
    print("All app routes:")
    for route in app.routes:
        print(f"  {route.path}")

except Exception as e:
    print(f"Error including AI strategy router: {e}")


