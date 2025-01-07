import numpy as np
from loguru import logger
from fastapi import APIRouter, Depends

from helpers import get_model
from data_types import FetalHealthData

router = APIRouter()

@router.get('/', tags=['Health'])
def home():
    return {'status': 'healthy'}

@router.post('/predict', tags=['Prediction'])
async def predict(request: FetalHealthData, model=Depends(get_model)):
    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)
    logger.info(received_data)
    prediction = model.predict(received_data)
    logger.info(prediction)
    return {"prediction": str(np.argmax(prediction[0]))}