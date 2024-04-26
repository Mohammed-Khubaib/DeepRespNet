from streamlit_elements import nivo
from streamlit_elements import elements, mui
def pieChart(DATA,kind,clr) :
    with elements(kind):
                with mui.Box(sx={"height": 400,"width":500}):
                    nivo.Pie(
                        data= DATA,
                        keys=["M","F"],
                        margin={ "top": 15, "right": 15, "bottom": 15, "left": 15 },
                        innerRadius=0.5,
                        padAngle=0.7,
                        cornerRadius=3,
                        sortByValue=True,
                        activeOuterRadiusOffset=8,
                        colors={ 'scheme': clr },
                        borderWidth=1,
                        borderColor = {
                            'from' : DATA,
                            'modifiers': 'darker'
                            # 'modifiers': [
                            #     ['darker', .6],
                            #     ['opacity', .5]
                            # ]
                        },
                        enableArcLinkLabels=False,
                        arcLinkLabelsSkipAngle=10,
                        arcLinkLabelsTextColor="#148237",
                        arcLinkLabelsThickness=5,
                        arcLinkLabelsColor={ "from": 'color' },
                        arcLabelsSkipAngle=2,
                        arcLabelsTextColor={
                        "from": 'color',
                        "modifiers": [
                            [
                                'darker',
                                3
                            ]
                        ]
                        },
                        defs = [
                        {
                            "id": "dots",
                            "type": "patternDots",
                            "background": "inherit",
                            "color": "rgba(61,213,109,255)",
                            "size": 4,
                            "padding": 1,
                            "stagger": "true"
                        },
                        {
                            "id": "lines",
                            "type": "patternLines",
                            "background": "inherit",
                            "color": "#148237",
                            "rotation": -45,
                            "lineWidth": 4,
                            "spacing": 10
                        },
                        {
                            "id": "lines1",
                            "type": "patternLines",
                            "background": "inherit",
                            "color": "red",
                            "rotation": -45,
                            "lineWidth": 4,
                            "spacing": 10
                        }
                        ],

                    fill = [
                        {
                            "match": {
                                "id": "M"
                            },
                            "id": "dots"
                        },
                        {
                            "match": {
                                "id": "F"
                            },
                            "id": "lines"
                        },
                        {
                            "match": {
                                "id": str(DATA[0]['id'])
                            },
                            "id": "dots"
                        },
                        {
                            "match": {
                                "id": str(DATA[1]['id'])
                            },
                            "id": "lines"
                        },
                    ],
                    legends=[
                                # {
                                #     "anchor": 'bottom',
                                #     "direction": 'row',
                                #     # "justify": "false",
                                #     "translateX": 1,
                                #     "translateY": 76,
                                #     "itemsSpacing": 10,
                                #     "itemWidth": 100,
                                #     "itemHeight": 18,
                                #     "itemTextColor": '#148237',
                                #     "itemDirection": 'left-to-right',
                                #     "itemOpacity": 2,
                                #     "symbolSize": 14,
                                #     "symbolShape": 'circle',
                                #     "effects": [
                                #         {
                                #             "on": 'hover',
                                #             "style": {
                                #                 "itemTextColor": '#63f08b'
                                #             }
                                #         }
                                #     ]
                                # }
                            ],
                            theme={
                            "background": "#ebf9ee",
                            "textColor": "#63f08b",
                            "tooltip": {
                                "container": {
                                    "background": "#eafaeb",
                                    "color": "#148237",
                                }
                            }
                        }
                    )