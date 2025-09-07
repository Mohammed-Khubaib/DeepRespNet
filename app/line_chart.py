import streamlit as st
from streamlit_elements import elements, mui
from streamlit_elements import nivo
import yaml
# Data = [{"x": i + 1, "y": loaded_history['accuracy'][i]} for i in range(800)]
# Define the IDs for which you want to create data entries
# ids = ['accuracy', 'val_accuracy']

# Generate the data using list comprehension
# Data = [
#     {
#         "id": id_value,
#         # "color": f"hsl({idx * 100}, 70%, 50%)",  # Generate a color based on index
#         "data": [
#             {"x": int(i + 1), "y": loaded_history[id_value][i]} for i in range(800)
#         ]
#     }
#     for idx, id_value in enumerate(ids)
# ]
# st.write(Data)
# st.write(Data[0]['color'])
def LineChart(Data):
    with elements("accuracy"):
        DATA = [
            {
                "id": Data[0]['id'],
                "color": "hsl(73, 70%, 50%)",
                # "color": "hsl(320, 70%, 50%)",
                "data": Data[0]['data']
            },
            {
                "id": Data[1]['id'],
                "color": "hsl(256, 70%, 50%)",
                # "color": "hsl(320, 70%, 50%)",
                "data": Data[1]['data']
            }
        ]
        with mui.Box(sx={"height": 400}):
            nivo.Line(
                data=DATA,
                margin={ "top": 25, "right": 110, "bottom": 110, "left": 60 },
                xScale={ "type": "linear" },
                yScale={
                    "type": "linear",
                    "min": 0.4,
                    "max": 1,
                    "stacked": False,
                    "reverse": False
                },
                # yFormat=">-.2f",
                axisTop= None,
                axisRight= None,
                axisBottom={
                    "tickSize": 10,
                    "tickPadding": 10,
                    "tickRotation": 0,
                    "legend": 'epoch',
                    "legendOffset": 50,
                    "legendPosition": 'middle'
                },
                axisLeft={
                    "tickSize": 10,
                    "tickPadding": 13,
                    "tickRotation": 10,
                    "legend": "Accuracy",
                    "legendOffset": -54,
                    "legendPosition": "middle"
                },
                enableGridX=False,
                enableGridY=False,
                # enableSlices="y",
                # colors={ 'scheme': 'accent' },
                # colors={ 'scheme': 'category10' },
                # colors={ 'scheme': 'nivo' },
                colors={ 'scheme': 'set2' },
                # enablePointLabel=True,
                # enablePoints = False,
                pointLabel="Accuracy",
                pointLabelYOffset=-24,
                lineWidth=1,
                pointSize=1,
                pointColor={ "theme": "background" },
                pointBorderWidth=1,
                pointBorderColor={ "from": "serieColor" },
                # pointLabelYOffset=-12,
                enableArea=True,
                 areaBaselineValue=0.4,
                areaOpacity=0.2,
                useMesh=True,
                # debugMesh=True,
                legends=[
                    {
                        "anchor": "bottom-right",
                        "direction": "column",
                        "justify": False,
                        "translateX": 100,
                        "translateY": 0,
                        "itemsSpacing": 0,
                        "itemDirection": "left-to-right",
                        "itemWidth": 80,
                        "itemHeight": 20,
                        "itemOpacity": 0.75,
                        "symbolSize": 12,
                        'symbolShape': "circle",
                        "symbolBorderColor": "'rgba(0, 0, 0, .5)",
                        "effects": [
                            {
                                "on": 'hover',
                                "style": {
                                    "itemBackground": 'green',
                                    "itemTextColor": '#63f08b',
                                    "itemOpacity": 1
                                }
                            }
                        ]
                    }
                ],
                theme={
                    "background": "#ebf9ee",
                    "textColor": "#148237",
                    "tooltip": {
                        "container": {
                            "background": "#fff",
                            "color": "#148237",
                        }
                    }
                }

                )
# yaml_file_path = "./model4accuracy.yaml"
# # Load the saved history data from the file
# with open(yaml_file_path, 'r') as file:
#     Data = yaml.load(file, Loader=yaml.FullLoader)
