import bentoml
from bentoml.io import File, JSON
from typing import Dict
from src.prediction.prediction import audio_preprocessing, deeprespnet_diagnosis_prediction
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="fs"
)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

CLASSES = ["Acute", "Chronic", "Healthy"]

@bentoml.service(resources={"cpu": "200m", "memory": "512Mi"})
class svc:
    def __init__(self):
        # Load model
        self.deeprespnet = bentoml.keras.load_model("lung_sound_classifier:latest")


    @bentoml.api
    def classify(self, file: Path) -> Dict:
        features = audio_preprocessing(file)
        predicted_class, confidence = deeprespnet_diagnosis_prediction(features=features,model=self.deeprespnet,use_bento_model=False)

        return {
            "predicted_class": str(predicted_class),
            "confidence": float(confidence)
        }
