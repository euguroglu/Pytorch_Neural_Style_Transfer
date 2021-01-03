from flask import Flask, render_template,Response,request,jsonify
from flask import redirect,url_for
import os
from PIL import Image
import numpy as np
import copy
import torch
import torch.optim as optim
from models import TransformerNet
from utils import *
from torchvision.utils import save_image
import tqdm
from torch.autograd import Variable

UPLOAD_FOLDER ='static/upload'


app = Flask(__name__)




@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        f=request.files['image']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path)

        style_img = path
        a = int(Image.open(style_img).size[0])
        b = int(Image.open(style_img).size[1])
        device = torch.device("cpu")

        transform = style_transform()

        transformer = TransformerNet().to(device)
        transformer.load_state_dict(torch.load("static/model/mosaic_10000.pth", map_location=torch.device('cpu')))
        transformer.eval()
    # Prepare input
        if a*b < 800000:
            image_tensor = Variable(transform(Image.open(style_img))).to(device)
            image_tensor = image_tensor.unsqueeze(0)
        else:
            image_tensor = Variable(transform(Image.open(style_img).resize((int(a/2),int(b/2))))).to(device)
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            stylized_image = denormalize(transformer(image_tensor)).cpu()

        save_image(stylized_image,"./static/predict/{}".format(filename))
        # stylized_image.save("./static/predict/{}".format(filename))
        #prediction pass to pipeline model

        return render_template("index.html",fileupload=True,img_name=filename)
    return render_template("index.html",fileupload=False)



if __name__ == '__main__':

    # app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    app.run(debug=True)
