# Databricks notebook source
# MAGIC %pip install gradio typing-extensions
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json
import gradio as gr

# COMMAND ----------

# MAGIC %run ../configs/00-config

# COMMAND ----------

# endpoint_url = f"""{os.environ['DATABRICKS_HOST']}/serving-endpoints/{serving_endpoint_name}/invocations"""
# endpoint_url

# COMMAND ----------

def get_response_data(question: str):
    
    return {
        "columns": ["messages"],
        "data": [[{"messages": [{"role": "user", "content": f"{question}"}]}]],
    }


def score_model(dataset: dict):
    
    url = os.environ.get("ENDPOINT_URL")
    headers = {
        "Authorization": f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        "Content-Type": "application/json",
    }
    ds_dict = {"dataframe_split": dataset}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()

# COMMAND ----------

greet(question="What is PetSmart?")

# COMMAND ----------

def greet(question: str):
    # filterdict={}
    # if not filter.strip() == '':
    #     filterdict={'Name':f'{filter}'}
    # dict = {'question':[f'{question}'], 'filter':[filterdict]}
    # assemble_question = pd.DataFrame.from_dict(dict)

    # data = score_model(assemble_question)
    request = get_response_data(question=question)
    data = score_model(dataset=request)

    answer = data['predictions'][0]['result']
    sources = list(set(data['predictions'][0]['sources']))
    sources = [source.replace(',', '\n') for source in sources]

    return [answer, sources]

def srcshowfn(chkbox):
    
    vis = True if chkbox==True else False
    print(vis)
    return gr.Textbox.update(visible=vis)

with gr.Blocks( theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.HTML(show_label=False, value="<img src='https://www.petsmart.com/on/demandware.static/Sites-PetSmart-Site/-/default/dw5aff61ba/images/petsmart-logo.png' height='30' width='30'/><div font size='1'>NRF</div>")
    with gr.Row():    
        gr.Markdown(
                """
            # PetSmart Puppy Bot
            This bot has been trained on PetSmart's product catalog, website blogs, and FAQs. For the purposes of this demo, we have only used data related to dogs. The fact sheets were transformed into embeddings and are used as a retriever for the model. Langchain was then used to compile the model, which is then hosted on Databricks MLflow. The application simply makes an API call to the model that's hosted in Databricks.
            """
            )
    with gr.Row():
        input = gr.Textbox(placeholder="ex. How can I transition my adult dog from a puppy food to an adult food?", label="Question")
        # inputfilter = gr.Textbox(placeholder="Dog Food", label="Filter (Optional)")
    with gr.Row():
        output = gr.Textbox(label="Prediction")
        # greet_btn = gr.Button("Respond", size="sm", scale=0).style(height=20)
        greet_btn = gr.Button("Respond", size="sm", scale=0)
    with gr.Row():
        srcshow = gr.Checkbox(value=False, label='Show sources')
    with gr.Row():
        outputsrc = gr.Textbox(label="Sources", visible=False)

    srcshow.change(srcshowfn, inputs=srcshow, outputs=outputsrc)
    greet_btn.click(fn=greet, inputs=[input], outputs=[output, outputsrc], api_name="greet")
    
demo.launch(share=True) 
