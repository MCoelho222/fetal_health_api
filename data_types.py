from pydantic import BaseModel

# Types for fetal health data
class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float