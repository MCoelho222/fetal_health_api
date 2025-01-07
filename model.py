import os
import mlflow
from loguru import logger

def load_model():
    logger.info('reading model...')
    MLFLOW_TRACKING_URI = 'https://dagshub.com/mcoelho5446/my-first-repo.mlflow'
    MLFLOW_TRACKING_USERNAME = 'mcoelho5446'
    MLFLOW_TRACKING_PASSWORD = '82f541e12472ccd9580c67f05b2da3680c830f9d'

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