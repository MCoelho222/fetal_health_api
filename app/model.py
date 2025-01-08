import os
import mlflow
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def load_model():
    logger.info('reading model...')
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
    MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

    logger.info('setting mlflow...')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info('creating client..')
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    logger.info('getting registered model...')
    registered_model = client.get_registered_model('fetal_health')

    logger.info('read model...')
    run_id = registered_model.latest_versions[-1].run_id

    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    logger.info(loaded_model)

    return loaded_model