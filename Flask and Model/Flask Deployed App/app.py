import json
import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


disease_info = pd.read_csv('disease_infoVN.csv' , encoding='utf-8')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)
# model.load_state_dict(torch.load("test1.pt"))
model.load_state_dict(torch.load("finaltest4.pt", map_location=torch.device('cpu')))
model.eval()


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/list', methods=['GET'])
def list():
    if request.method == 'GET':
        json_data = disease_info.to_dict(orient='records')
        json_array = json.dumps(json_data)
        return json_array

@app.route('/cate', methods=['GET'])
def cate():
    if request.method == 'GET':
        name = request.args.get('name')  # Get the value of the 'name' query parameter

        if name:
            cate_df_filter = disease_info.loc[disease_info['cate'] == name].drop('cate_img', axis=1)
        else:
            cate_df = disease_info.dropna()
            cate_df_filter = cate_df[['index', 'cate', 'cate_img']]


        # response = cate_df_filter.to_json(orient='records')[1:-1].replace('},{', '} {')
        json_data = cate_df_filter.to_dict(orient='records')
        json_array = json.dumps(json_data)

        return json_array




@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        print(request)
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        cate = disease_info['cate'][pred]
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible_steps'][pred]
        image_url = disease_info['image_url'][pred]
        # supplement_name = supplement_info['supplement name'][pred]
        # supplement_image_url = supplement_info['supplement image'][pred]
        # supplement_buy_link = supplement_info['buy link'][pred]
        rs ={
        "cate":cate,
        "disease_name": title,
        "description": description,
        "Possible_steps": prevent,
        "image_url": image_url,



        }
        return rs
        # return render_template('submit.html' , title = title , desc = description , prevent = prevent ,
        #                        image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)


if __name__ == '__main__':
    app.run(debug=True)
