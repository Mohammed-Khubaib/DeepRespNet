import bentoml
from bentoml.io import File, JSON
from typing import Dict
from src.prediction.prediction import audio_preprocessing, deeprespnet_diagnosis_prediction
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="fs"
)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import fs

# Load model
deeprespnet = bentoml.keras.get("lung_sound_classifier:latest").to_runner()

svc = bentoml.Service("deeprespnet", runners=[deeprespnet])

CLASSES = ["Acute", "Chronic", "Healthy"]

@svc.api(input=File(), output=JSON())
def classify(file) -> Dict:
    features = audio_preprocessing(file)
    predicted_class, confidence = deeprespnet_diagnosis_prediction(features=features,model=deeprespnet,use_bento_model=True)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }
