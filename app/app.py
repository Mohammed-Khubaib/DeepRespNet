import streamlit as st
import json
from streamlit_lottie import st_lottie
from st_on_hover_tabs import on_hover_tabs
from st_audiorec import st_audiorec
import matplotlib.pyplot as plt
import numpy as np
import yaml
import librosa
import librosa.display
from line_chart import LineChart
from pie_chart import pieChart
import requests
import os

# --- Configuration with environment variables for Docker ---
# FastAPI Server 
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "api")  # Docker service name
FASTAPI_PORT = os.getenv("FASTAPI_PORT", "8000")
server_health = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/healthz"
create_new_api_key = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/create-key"
verify_api_key = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/security_check"

# BentoML Server
BENTOML_HOST = os.getenv("BENTOML_HOST", "bento_ml")  # Docker service name
BENTOML_PORT = os.getenv("BENTOML_PORT", "3000")
model_server_health = f"http://{BENTOML_HOST}:{BENTOML_PORT}/healthz"
model_inference = f"http://{BENTOML_HOST}:{BENTOML_PORT}/classify"


# # --- Uncoment for local development, but make sure to run the fastapi server (api_key) and (bentoml) ---
# # --- Configuration with environment variables for Local Development ---
# # --- FastAPI Server
# server_health = "http://localhost:8000/healthz"
# create_new_api_key = "http://localhost:8000/create-key"
# verify_api_key = "http://localhost:8000/security_check"
# # --- Bentoml Server
# model_server_health = "http://localhost:3000/healthz"
# model_inference = "http://localhost:3000/classify"
# # ---

yaml_file_path = "./model5accuracy.yaml"

st.set_page_config(page_title="DeepRespNet", page_icon='🫁', layout="wide",initial_sidebar_state='auto')
def load_lottie_file(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)

# Hide the "Made with Streamlit" footer
# Define a CSS style for the text
hide_streamlit_style="""
    <style>
    #MainMenu{visibility:hidden;}
    footer{visibility:hidden;}
    h1 {
        color: #01FFB3 ;
    }
    h2 {
        color: #01FFB3;
    }
    h3 {
        color: #12FFE2;
    }
    button:hover {
    background-color: green;
    }
    /* The progress bars */
        .stProgress > div > div > div > div {
            background: linear-gradient(to right, #00EEFF, #01FFB3);
            border-radius: 10px;
        }
        /* The text inside the progress bars */
        .stProgress > div > div > div > div > div {
            color: white;
        }
    </style>
    """
# st.markdown(hide_streamlit_style,unsafe_allow_html=True)
lottie_file1 = load_lottie_file('./animations/SoundRecordingAnimation.json')
lottie_file2 = load_lottie_file('./animations/Lungs.json')
lottie_file3 = load_lottie_file('./animations/dashboard animation.json')
lottie_file4 = load_lottie_file('./animations/Stethoscope.json')
tabs = st.empty()
with st.sidebar:
        cp,cq,cr= st.columns([0.2,0.6,0.2])
        with cq:
            st_lottie(lottie_file2,speed=1,reverse=False,height=150,width=150)
        st.latex(r'''\large\color{green}\textbf{DeepRespNet}''')
        st.divider()
        tabs = on_hover_tabs(tabName=['Dashboard','Auscultation','API Key','Prediction'], 
                         iconName=['monitor_heart','radio_button_checked','api','batch_prediction'], default_choice=0,
                         styles={'navtab': {'background-color':'#fff1',
                                            'color': '#357142',
                                            'font-size': '18px',
                                            'transition': '.3s',
                                            'white-space': 'nowrap',
                                            'text-transform': 'uppercase'},
                                 'tabOptionsStyle': {':hover :hover': {'color': '#3dd56d',
                                                                     'cursor': 'pointer'}},
                             },
    )
st.latex(r'''\Huge\color{green}\textbf{DeepRespNet}''')
c1,c2,c3 = st.columns([0.3,0.6,0.1])
if tabs == 'Dashboard':
    st.success('### :green[A Deep Learning Approach for Potential Classification of Respiratory Diseases]')
    # st.warning(":rainbow[Hello World]")
    listTabs = [
        '$$\large\\textbf{About}$$',
        '$$\large\\textbf{Instructions}$$',
        '$$\large\\textbf{Dataset}$$',
        '$$\large\\textbf{Model}$$',
        ]
    whitespace = 2
    tab_labels = [s.center(len(s) + whitespace, "\u2001") for s in listTabs]
    sections = st.tabs(tab_labels)
    with sections[0]:
        ca,cb = st.columns([0.5,0.5])
        with ca:
            st_lottie(lottie_file3,speed=1,reverse=False,height=350,width=350)
        with cb:
            st.success('##### :green[:green[DeepRespNet is a Deep Learning Model designed to analyze respiratory sounds to determine the state of your respiratory health. Using Deep learning techniques, our Model classifies sounds into categories of] '+r'$$\small\color{darkorange}\textit{health}$$'+ ' :green[,] '+r'''$$\small\color{darkorange}\textit{chronic disease}$$'''+ ' :green[, and] '+r'''$$\small\color{darkorange}\textit{acute condition}$$'''+' :green[DeepRespNet Web application provides an easy-to-use platform for individuals and healthcare professionals alike to assess respiratory health through audio recordings. Our goal is to aid in the early detection of respiratory conditions, supporting timely and accurate medical intervention.]]')

            st.write('\n\n')
            st.write('\n\n')
            st.write('\n\n')
            st.write('\n\n')
            st.write('\n\n')
    with sections[1]:
        st.warning("⚠️ There is a sidebar in this Webapp on the left hand side")
        st.error("There are two main stages of DeepRespNet web app:"+"\n 1. Auscultation \n 2. Prediction")
            
        st.subheader(":green[Digital Lung Auscultation:]",anchor=False)
        c1,c2 = st.columns([0.5,0.5])
        with c1:
            st.info("Step 1: From the sidebar, navigate to 'Auscultation'")
            st.info("Step 2: connect the Digital Stethoscope")
            st.info("Step 3: press "+r'''$$\color{orange} \text{start} $$'''+' to begin the Auscultation process')
            st.info("Step 4: press "+r'''$$\color{red} \text{stop} $$'''+' to end the Auscultation process')
            st.success("Step 5: Once the audio is ready to be downloaded, press "+r'''$$\color{green} \text{Downloade} $$'''+" and save the audio")
        with c2:
            st_lottie(lottie_file4,speed=1,reverse=False,height=400,width=400)

        st.warning(r'''⚠️$$\textbf{ Note : To get a reasonable result}: \\$$'''+'''\n - place the Digital Stethoscope directly on your lungs \n - breath deeply in and out through your mouth for at least 15 seconds \n''')
            
        st.divider()
            
        st.write("### Classification:")
        st.info("Step 1: Upload the recorded Auscultation audio for Analysis")
        st.success("- An Artificial Neural Network evaluates the recorded Auscultation audio. \n \n - The Results are promptly displayed within seconds of uploading the audio")
    with sections[2]:
        # Load the saved history data from the file
        with open(yaml_file_path, 'r') as file:
            Data = yaml.load(file, Loader=yaml.FullLoader)
        DATA1 = [
                    {"id": "M", "label": "Male", "value": int(79),"color": "#fff"},
                    {"id": "F", "label": "Female", "value": int(46), "color": "#fff)"}
                ]
        DATA2 = [
            {'id':'under weight','label':'underweight','value':int(3)},
            {'id':'normal','label':'normal','value':int(19)},
            {'id':'overweigh','label':'over weight','value':int(38)},
            {'id':'obese','label':'obese','value':int(15)},
            {'id':'Under Weight','label':'Under Weight','value':int(32)},
            {'id':'Normal','label':'Normal','value':int(8)},
            {'id':'Over Weight','label':'Over Weight','value':int(8)},

        ]
        DATA3 = [
            {'id':'COPD','label':'COPD','value':int(51)},
            {'id':'Healthy','label':'Healthy','value':int(21)},
            {'id':'URTI','label':'URTI','value':'11%'},
            {'id':'Bronchiectasis','label':'Bronchiectasis','value':int(6)},
            {'id':'Bronchiolitis','label':'Bronchiolitis','value':int(5)},
            {'id':'LRTI','label':'LRTI','value':int(5)},
            {'id':'Asthma','label':'Asthma','value':1},
        ]

        c1,c2 = st.columns([0.5,0.5])
        with c1:
            st.success("##### :green[Gender]")
            pieChart(DATA=DATA1,kind='gender',clr='greens')
            st.success("##### :green[Adults BMI]")
            pieChart(DATA=DATA2[:4],kind='ABMI',clr='greens')
        with c2:
            st.success("##### :green[Classes Distribution]")
            pieChart(DATA=DATA3,kind='Class',clr='greens')
            st.success("##### :green[Child BMI]")
            pieChart(DATA=DATA2[4:],kind='CBMI',clr='greens')
    with sections[3]:
        st.success("##### :green[DeepRespNet Architecture]")
        st.image('./animations/DeepRespNet Architecture.png')
        st.success("##### :green[Accuracy vs Validation Accuracy]")
        LineChart(Data=Data)

elif tabs =='Auscultation':
    with c2:
        ca,cb,cc = st.columns([0.2,0.5,0.2])

        with cb:
            st_lottie(lottie_file1,speed=1,reverse=False,height=150,width=150,quality='high')
        wav_audio_data = st_audiorec()

elif tabs =='API Key':
    try:
        server_health_response = requests.get(url=server_health)
        if server_health_response.status_code == 200:
            st.success("✅ Connection established successfully!")
            try:
               owner = st.text_input("Enter the name of the Owner who will own this API key")
               params = {"owner": owner}
               if(owner!=""):
                response = requests.post(create_new_api_key, params=params)
                if response.status_code == 200:
                        api_key = response.json().get("api_key")
                        st.header("[✅] API Key generated: ")
                        st.code(api_key)
                else:
                        st.error(f"[❌] Failed to create API key: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ We are facing some technical issues: {e}")
    except requests.exceptions.RequestException as e:
        # st.error(f"❌ Error during health check: {e}")
        st.error("❌ Error during health check: 😔")

else :
    try:
        server_health_response = requests.get(url=server_health)
        if server_health_response.status_code == 200:
            st.success("✅ Connection established successfully!")
            try:
                api_key = st.text_input("Provide An API key to access the model")
                if api_key != "":
                    headers = {
                        "X-API-Key": api_key
                    }
                    response = requests.get(verify_api_key, headers=headers)
                    if(response.status_code == 200):
                        result = response.json()
                        st.success(result['message'])
                        try:
                            model_health_response = requests.get(url=model_server_health)
                            if model_health_response.status_code == 200:
                                st.success("✅ Connection established successfully!")
                                try:
                                    # Try making a GET request (or use POST if your endpoint requires it)

                                    audio_file = st.file_uploader('Upload the lung sound',type='.wav',accept_multiple_files=False)
                                    predict = st.empty()
                                    if (audio_file is not None):
                                        st.audio(audio_file)
                                        ca,cb,cc = st.columns([0.5,0.2,0.4])
                                        files = {'file': (audio_file.name, audio_file, 'audio/wav')}
                                        headers = {'accept': 'application/json'}
                                        with cb:
                                            predict = st.button("predict")
                                        if predict:
                                            response = requests.post(url=model_inference, files=files, headers=headers)
                                            if response.status_code == 200:
                                                result = response.json()
                                                cls= result['predicted_class']
                                                conf= result['confidence']
                                                # st.info(f'{cls}')
                                                if cls == 'Chronic':
                                                    # st.error(f':red[{cls} : {round(conf*100,3)}]%')
                                                    # st.success(f"- There is a higher probabilty of about {round(conf*100,3)}% that your respiratory system is effected by {cls} lung diseases \n \n - Please remember that this is not a medical diagnosis. \n \n - if in doubt,it is best to seek a doctor's opinion")
                                                    confidence=round(conf*100,3)
                                                    st.error(r'''⚠️$$\textbf{ Result }: \\$$'''+ f"\n\n - There is a higher probabilty of about "+f''':red[{confidence}%]'''+f" that your respiratory system is effected by {cls} lung diseases \n \n - This percentage indicates the confidence level of the analysis based on the provided respiratory sound data. \n \n - Please Remember, this isn't a medical diagnosis. \n \n - If unsure, it is best to seek a doctor's opinion")
                                                elif cls == 'Healthy':
                                                    # st.success(f':green[{cls} : {round(conf*100,3)}]')
                                                    confidence=round(conf*100,3)
                                                    st.success(r'''⚠️$$\textbf{ Result }: \\$$'''+ f"\n\n - Great news! Your respiratory system appears to be healthy.\n\n - The assessment suggests that your respiratory system is unaffected by lung diseases, with a confidence level of "+f':violet[{confidence}%]'+f" \n - This percentage indicates the confidence level of the analysis based on the provided respiratory sound data. \n \n \n - Please Remember, this isn't a medical diagnosis. \n \n - If unsure, it is best to seek a doctor's opinion")
                                                else:
                                                    # st.warning(f':orange[{cls} : {round(conf*100,3)}]%')
                                                    confidence=round(conf*100,3)
                                                    # st.warning(r'''⚠️$$\textbf{ Result }: \\$$'''+ f"\n\n - There is a higher probabilty of about {round(conf*100,3)}% that your respiratory system is effected by {cls} lung diseases \n \n - Please Remember, this isn't a medical diagnosis. \n \n - If unsure, it is best to seek a doctor's opinion")
                                                    st.warning(r'''⚠️$$\textbf{ Result }: \\$$'''+ f"\n\n - There is a higher probabilty of about "+f':orange[{confidence}%]'+f" that your respiratory system is effected by {cls} lung diseases \n - This percentage indicates the confidence level of the analysis based on the provided respiratory sound data. \n \n \n - Please Remember, this isn't a medical diagnosis. \n \n - If unsure, it is best to seek a doctor's opinion")
                                            else:
                                                st.warning(f"⚠️ Server responded with status code: {response.status_code}")
                                                st.warning(f"Response: {response.text}")
                                        else:
                                            # Load the audio file
                                            y, sr = librosa.load(audio_file)
                                        
                                            # Generate mel spectrogram
                                            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)  # Increased resolution
                                            S_dB = librosa.amplitude_to_db(S, ref=np.max)
                                            
                                            # Plotting the mel spectrogram
                                            fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted size for better visibility
                                            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='log', ax=ax)
                                            ax.set_title('Mel-frequency Spectrogram', fontsize=24)
                                            ax.set_xlabel('Time (s)', fontsize=16)
                                            ax.set_ylabel('Frequency (Hz)', fontsize=16)
                                            fig.colorbar(img, ax=ax, format='%+2.0f dB')
                                            
                                            # Display the plot in Streamlit
                                            st.pyplot(fig)
                                        
                                except requests.exceptions.ConnectionError:
                                    st.error("❌ Failed to connect. Is the BentoML server running?")
                                except requests.exceptions.Timeout:
                                    st.error("⏱️ Request timed out.")
                                except requests.exceptions.RequestException as e:
                                    st.error(f"⚠️ An error occurred: {e}")
                            else:
                                st.error(f"⚠️ Health check failed with status code {model_health_response.status_code}")

                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ We are facing some technical issues: {e}")
                    else:
                        st.warning("Please Provide an Valid API Key")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ We are facing some technical issues: {e}")
    
    except requests.exceptions.RequestException as e:
        # st.error(f"❌ Error during health check: {e}")
        st.error("❌ Error during health check: 😔")