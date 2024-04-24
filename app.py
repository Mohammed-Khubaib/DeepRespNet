import streamlit as st
import json
from streamlit_lottie import st_lottie
from st_on_hover_tabs import on_hover_tabs
from st_audiorec import st_audiorec
import matplotlib.pyplot as plt
from IPython.display import Audio
import numpy as np
import librosa
import librosa.display
from tensorflow.keras.models import load_model

def load_lottie_file(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)

def stretch(data, rate):
    data = librosa.effects.time_stretch(y=data, rate=rate)
    return data
classes = ["Chronic" ,"Acute ", "Healthy"]
def gru_prediction(audio):
    data_x, sampling_rate = librosa.load(audio)
    data_x = stretch (data_x,1.2)

    features = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=52).T,axis = 0)

    features = features.reshape(1,52)

    test_pred = loaded_model.predict(np.expand_dims(features, axis = 1))
    classpreds = str(classes[np.argmax(test_pred[0], axis=1)[0]])
    confidence = round(float(test_pred.T[test_pred[0].mean(axis=0).argmax()].mean()),3)
    return classpreds , confidence

st.set_page_config(page_title="DeepRespNet", page_icon='🫁', layout="wide",initial_sidebar_state='auto')
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
tabs = st.empty()
with st.sidebar:
        cp,cq,cr= st.columns([0.2,0.6,0.2])
        with cq:
            st_lottie(lottie_file2,speed=0.5,reverse=False,height=150,width=150)
        st.latex(r'''\large\color{LimeGreen}\textbf{DeepRespNet}''')
        st.divider()
        tabs = on_hover_tabs(tabName=['Dashboard','Auscultation','Upload'], 
                         iconName=['monitor_heart','radio_button_checked','save'], default_choice=0,
                         styles={'navtab': {'background-color':'#fff1',
                                            'color': '#818181',
                                            'font-size': '18px',
                                            'transition': '.3s',
                                            'white-space': 'nowrap',
                                            'text-transform': 'uppercase'},
                                 'tabOptionsStyle': {':hover :hover': {'color': '#3dd56d',
                                                                     'cursor': 'pointer'}},
                             },
    )
st.latex(r'''\Huge\color{LimeGreen}\textbf{DeepRespNet}''')
c1,c2,c3 = st.columns([0.3,0.6,0.1])
if tabs == 'Dashboard':
    st.latex(r'''\color{green}\textbf{A Deep Learning Approach for Potential Classification of Respiratory Diseases}''')
    st.warning("Hello World")
    with st.expander(r'''$$\color{green}\large\textbf{Hello 👋! Follow the steps below to begin.}$$'''):
            st.warning("⚠️ There is a sidebar in this Webapp on the left hand side")
            st.error("There are two main stages of DeepRespNet web app:")
            
            st.subheader(":green[Digital Lung Auscultation:]",anchor=False)
            st.info("Step 1: From the sidebar, navigate to 'Auscultation'")
            st.info("Step 2: connect the Digital Stethoscope")
            st.info("Step 3: press "+r'''$$\color{orange} \text{start} $$'''+' to begin the Auscultation process')
            st.info("Step 4: press "+r'''$$\color{red} \text{stop} $$'''+' to end the Auscultation process')
            st.success("Step 5: Once the audio is ready to be downloaded, "+r'''$$\color{green} \text{Downloaded} $$'''+" and save the audio")
            
            st.divider()
            
            st.write("### Data Processing:")
            st.info("From the sidebar, navigate to 'Data Processing'")
            st.info("Step 1: Upload the file you downloaded from the 'Download Results' page")
            st.success("Step 2: A new processed file will be ready for you to download. Press 'Download' and save it as well")
            
            st.divider()
            
            st.write("### Analysis:")
            st.info("Upload the preprocessed file for Analysis")
            st.success("Fully Automated Result Analysis is performed")
elif tabs =='Auscultation':
    with c2:
        ca,cb,cc = st.columns([0.2,0.5,0.2])

        with cb:
            st_lottie(lottie_file1,speed=0.5,reverse=False,height=150,width=150,quality='high')
        wav_audio_data = st_audiorec()

else :
    loaded_model = load_model("./diagnosis_GRU_CNN_1.h5")
    if load_model is not None:
        st.success("Model Loaded")
    audio_file = st.file_uploader('Upload the lung sound',type='.wav',accept_multiple_files=False)
    ca,cb,cc = st.columns([0.5,0.2,0.4])
    predict = st.empty()
    if (audio_file is not None):
        with cb:
            predict = st.button("predict")
        if predict:
            cls, conf = gru_prediction(audio_file)
            # st.info(f'{cls}')
            st.audio(audio_file)
            if cls == 'Chronic':
                st.error(f':red[{cls} : {conf}]')
            elif cls == 'Healthy':
                st.success(f':green[{cls} : {conf}]')
            else:
                # st.warning(f'{cls}')
                st.warning(f':orange[{cls} : {conf}]')
