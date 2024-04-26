import streamlit as st
import json
from streamlit_lottie import st_lottie
from st_on_hover_tabs import on_hover_tabs
from st_audiorec import st_audiorec
import matplotlib.pyplot as plt
from IPython.display import Audio
import numpy as np
import yaml
import librosa
import librosa.display
from tensorflow.keras.models import load_model
st.set_page_config(page_title="DeepRespNet", page_icon='🫁', layout="wide",initial_sidebar_state='auto')
from linechart import LineChart
from pieChart import pieChart
yaml_file_path = "./model4accuracy.yaml"
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
            st_lottie(lottie_file2,speed=0.5,reverse=False,height=150,width=150)
        st.latex(r'''\large\color{green}\textbf{DeepRespNet}''')
        st.divider()
        tabs = on_hover_tabs(tabName=['Dashboard','Auscultation','Prediction'], 
                         iconName=['monitor_heart','radio_button_checked','batch_prediction'], default_choice=0,
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
            st_lottie(lottie_file3,speed=0.5,reverse=False,height=350,width=350)
        with cb:
            st.write('\n\n')
            st.write('\n\n')
            st.write('\n\n')
            st.write('\n\n')
            st.write('\n\n')
            st.success('##### :green[:green[This project is built to determine if your respiratory system is ]'+r'$$\small\color{darkorange}\textit{ healthy}$$'+' :green[or affected by disease states, such as] '+ r'''$$\small\color{darkorange}\textit{Chronic}$$'''+ ' :green[or] '+r'''$$\small\color{darkorange}\textit{acute}$$'''+' :green[conditions.]]')
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
            st_lottie(lottie_file4,speed=0.5,reverse=False,height=400,width=400)

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
        st.success("##### :green[Accuracy]")
        LineChart(Data=Data)

elif tabs =='Auscultation':
    with c2:
        ca,cb,cc = st.columns([0.2,0.5,0.2])

        with cb:
            st_lottie(lottie_file1,speed=0.5,reverse=False,height=150,width=150,quality='high')
        wav_audio_data = st_audiorec()

else :
    loaded_model = load_model("./diagnosis_GRU_CNN_4.h5")
    if load_model is not None:
        st.success(":green[Model Loaded]")
    audio_file = st.file_uploader('Upload the lung sound',type='.wav',accept_multiple_files=False)
    st.audio(audio_file)
    ca,cb,cc = st.columns([0.5,0.2,0.4])
    predict = st.empty()
    if (audio_file is not None):
        with cb:
            predict = st.button("predict")
        if predict:
            cls, conf = gru_prediction(audio_file)
            # st.info(f'{cls}')
            if cls == 'Chronic':
                # st.error(f':red[{cls} : {round(conf*100,3)}]%')
                # st.success(f"- There is a higher probabilty of about {round(conf*100,3)}% that your respiratory system is effected by {cls} lung diseases \n \n - Please remember that this is not a medical diagnosis. \n \n - if in doubt,it is best to seek a doctor's opinion")
                confidence=round(conf*100,3)
                st.error(r'''⚠️$$\textbf{ Result }: \\$$'''+ f"\n\n - There is a higher probabilty of about "+f''':red[{confidence}%]'''+f" that your respiratory system is effected by {cls} lung diseases \n \n - Please Remember, this isn't a medical diagnosis. \n \n - If unsure, it is best to seek a doctor's opinion")
            elif cls == 'Healthy':
                # st.success(f':green[{cls} : {round(conf*100,3)}]')
                confidence=round(conf*100,3)
                st.success(r'''⚠️$$\textbf{ Result }: \\$$'''+ f"\n\n - Great news! Your respiratory system appears to be healthy.\n\n - The assessment suggests that your respiratory system is unaffected by lung diseases, with a confidence level of "+f':green[{confidence}%]'+f" \n \n - Please Remember, this isn't a medical diagnosis. \n \n - If unsure, it is best to seek a doctor's opinion")
            else:
                # st.warning(f':orange[{cls} : {round(conf*100,3)}]%')
                confidence=round(conf*100,3)
                # st.warning(r'''⚠️$$\textbf{ Result }: \\$$'''+ f"\n\n - There is a higher probabilty of about {round(conf*100,3)}% that your respiratory system is effected by {cls} lung diseases \n \n - Please Remember, this isn't a medical diagnosis. \n \n - If unsure, it is best to seek a doctor's opinion")
                st.warning(r'''⚠️$$\textbf{ Result }: \\$$'''+ f"\n\n - There is a higher probabilty of about "+f':orange[{confidence}%]'+f" that your respiratory system is effected by {cls} lung diseases \n \n - Please Remember, this isn't a medical diagnosis. \n \n - If unsure, it is best to seek a doctor's opinion")
