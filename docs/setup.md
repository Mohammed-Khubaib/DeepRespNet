---
icon: material/rocket-launch-outline
title: setup
social:
  cards_layout_options:
    title: Documentation that simply works
---

## :file_folder: Project Structure 


```bash
.
├── src # source code
├── data # data folder (https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)
├── notebooks # DeepRespNet Models development Notebooks
├── service.py # required by bentoml
├── bentofile.yaml # required by bentoml
├── requirements.txt # required by bentoml
├── pyproject.toml # required by uv
└── archive # old code
```
    
---

### 1. Installation
1. Python Environment Setup with UV

    **Install UV** (if not already installed): 
        [Astral UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)

2. Clone the repository:
   ```console
    $ git clone https://github.com/Mohammed-Khubaib/DeepRespNet.git

    # Navigate to the project directory:
    $ cd DeepRespNet
   ```

3. Setting up UV (Python's Package Manager):

    === "macOS"

        ```console
        # Install dependencies
        $ uv sync

        # Activate environment (Linux/macOS)
        $ source .venv/bin/activate
        ```

    === "Linux"

        ```console
        # Install dependencies
        $ uv sync

        # Activate environment (Linux/macOS)
        $ source .venv/bin/activate
        ```
4. Dataset Source
    - The dataset used in this project is the [Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) available on Kaggle.

- Run The Application locally:
    - Run API(FastAPI)
        ```console
        $ uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
        ```
    - Run Streamlit application 
        ```console
        $ uv run streamlit run main.py
        ```
---

### 2. bentoml server setup