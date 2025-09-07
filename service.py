import bentoml
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
        self.deeprespnet = bentoml.mlflow.load_model("peepseek:latest")


    @bentoml.api
    def classify(self, file: Path) -> Dict:
        """
            Classify an audio file to predict respiratory condition.

            This endpoint accepts the path to an audio file, performs preprocessing,
            and uses a trained DeepRespNet model to predict the respiratory condition.
            
            ### Request Parameters:
            - **file** (`Path`): Path to the input audio file (e.g., `.wav` format).

            ### Response:
            Returns a JSON object with the predicted class and its associated confidence score.

            #### Example response:
            ```json
            {
            "predicted_class": "Crackles",
            "confidence": 0.92
            }
            ```

            ### Returns:
            - **dict**: A dictionary with:
                - `predicted_class` (`str`): The label predicted by the model.
                - `confidence` (`float`): The confidence score of the prediction (between 0 and 1).
        """
        features = audio_preprocessing(file)
        predicted_class, confidence = deeprespnet_diagnosis_prediction(features=features,model=self.deeprespnet,use_bento_model=False)

        return {
            "predicted_class": str(predicted_class),
            "confidence": float(confidence)
        }