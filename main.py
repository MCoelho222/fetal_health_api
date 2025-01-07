from fastapi import FastAPI
from model import load_model
from contextlib import asynccontextmanager

from endpoints import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded_model = load_model()  # Consider using app.state to store the model
    yield

app = FastAPI(
    title='Fetal Health API',
    openapi_tags=[
        {
            'name': 'Health',
            'description': 'Get Api health'
        },
        {
            'name': 'Prediction',
            'description': 'Model prediction'
        },
    ],
    lifespan=lifespan
)

# Include the router
app.include_router(router)


