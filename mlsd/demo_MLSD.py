'''
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0
'''
# for demo
import os
from flask import Flask, request, session, json, Response, render_template, abort, send_from_directory
import requests
from urllib.request import urlopen
from io import BytesIO
import uuid
import cv2
import time
import argparse

# for tflite
import numpy as np
from PIL import Image
import tensorflow as tf

# for square detector
from utils import pred_squares

os.environ['CUDA_VISIBLE_DEVICES'] = '' # CPU mode

# flask
app = Flask(__name__)
logger = app.logger
logger.info('init demo app')

# config
parser = argparse.ArgumentParser()

## model parameters
parser.add_argument('--tflite_path', default='./tflite_models/M-LSD_512_large_fp16.tflite', type=str)
parser.add_argument('--input_size', default=512, type=int,
                    help='The size of input images.')

## LSD parameter
parser.add_argument('--score_thr', default=0.10, type=float,
                    help='Discard center points when the score < score_thr.')

## intersection point parameters
parser.add_argument('--outside_ratio', default=0.10, type=float,
                    help='''Discard an intersection point 
                    when it is located outside a line segment farther than line_length * outside_ratio.''')
parser.add_argument('--inside_ratio', default=0.50, type=float,
                    help='''Discard an intersection point
                    when it is located inside a line segment farther than line_length * inside_ratio.''')

## ranking boxes parameters
parser.add_argument('--w_overlap', default=0.0, type=float,
                    help='''When increasing w_overlap, the final box tends to overlap with
                    the detected line segments as much as possible.''')
parser.add_argument('--w_degree', default=1.14, type=float,
                    help='''When increasing w_degree, the final box tends to be
                    a parallel quadrilateral with reference to the angle of the box.''')
parser.add_argument('--w_length', default=0.03, type=float,
                    help='''When increasing w_length, the final box tends to be
                    a parallel quadrilateral with reference to the length of the box.''')
parser.add_argument('--w_area', default=1.84, type=float,
                    help='When increasing w_area, the final box tends to be the largest one out of candidates.')
parser.add_argument('--w_center', default=1.46, type=float,
                    help='When increasing w_center, the final box tends to be located in the center of input image.')

## flask demo parameter
parser.add_argument('--port', default=5000, type=int,
                    help='flask demo will be running on http://0.0.0.0:port/')


class model_graph:
    def __init__(self, args):
        self.interpreter, self.input_details, self.output_details = self.load_tflite(args.tflite_path)
        self.params = {'score': args.score_thr,'outside_ratio': args.outside_ratio,'inside_ratio': args.inside_ratio, 
                       'w_overlap': args.w_overlap,'w_degree': args.w_degree,'w_length': args.w_length,
                       'w_area': args.w_area,'w_center': args.w_center}
        self.args = args


    def load_tflite(self, tflite_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        return interpreter, input_details, output_details


    def pred_tflite(self, image):
        segments, squares, score_array, inter_points = pred_squares(image, self.interpreter, self.input_details, self.output_details, [self.args.input_size, self.args.input_size], params=self.params)

        output = {}
        output['segments'] = segments
        output['squares'] = squares
        output['scores'] = score_array
        output['inter_points'] = inter_points

        return output


    def read_image(self, image_url):
        response = requests.get(image_url, stream=True)
        image = np.asarray(Image.open(BytesIO(response.content)).convert('RGB'))
        
        max_len = 1024
        h, w, _ = image.shape
        org_shape = [h, w]
        max_idx = np.argmax(org_shape)

        max_val = org_shape[max_idx]
        if max_val  > max_len:
            min_idx = (max_idx + 1) % 2
            ratio = max_len / max_val
            new_min = org_shape[min_idx] * ratio
            new_shape = [0, 0]
            new_shape[max_idx] = 1024
            new_shape[min_idx] = new_min

            image = cv2.resize(image, (int(new_shape[1]), int(new_shape[0])), interpolation=cv2.INTER_AREA)

        return image
    

    def init_resize_image(self, im, maximum_size=1024):
        h, w, _ = im.shape
        size = [h, w]
        max_arg = np.argmax(size)
        max_len = size[max_arg]
        min_arg = max_arg - 1
        min_len = size[min_arg]
        if max_len < maximum_size:
            return im
        else:
            ratio = maximum_size / max_len
            max_len = max_len * ratio
            min_len = min_len * ratio
            size[max_arg] = int(max_len)
            size[min_arg] = int(min_len)

            im = cv2.resize(im, (size[1], size[0]), interpolation = cv2.INTER_AREA)

            return im


    def decode_image(self, session_id, rawimg):
        dirpath = os.path.join('static/results', session_id)

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        save_path = os.path.join(dirpath, 'input.png')
        input_image_url = os.path.join(dirpath, 'input.png')
    
        img = cv2.imdecode(np.frombuffer(rawimg, dtype='uint8'), 1)[:,:,::-1]
        img = self.init_resize_image(img)
        cv2.imwrite(save_path, img[:,:,::-1])

        return img, input_image_url


    def draw_output(self, image, output, save_path='test.png'):
        color_dict = {'red': [255, 0, 0],
                      'green': [0, 255, 0],
                      'blue': [0, 0, 255],
                      'cyan': [0, 255, 255],
                      'black': [0, 0, 0],
                      'yellow': [255, 255, 0],
                      'dark_yellow': [200, 200, 0]}
        
        line_image = image.copy()
        square_image = image.copy()
        square_candidate_image = image.copy()
        
        line_thick = 5

        # output > line array
        for line in output['segments']:
            x_start, y_start, x_end, y_end = [int(val) for val in line]
            cv2.line(line_image, (x_start, y_start), (x_end, y_end), color_dict['red'], line_thick)
        
        inter_image = line_image.copy()
               
        for pt in output['inter_points']:
            x, y = [int(val) for val in pt]
            cv2.circle(inter_image, (x, y), 10, color_dict['blue'], -1)

        for square in output['squares']:
            cv2.polylines(square_candidate_image, [square.reshape([-1, 1, 2])], True, color_dict['dark_yellow'], line_thick)

        for square in output['squares'][0:1]:
            cv2.polylines(square_image, [square.reshape([-1, 1, 2])], True, color_dict['yellow'], line_thick)
            for pt in square:
                cv2.circle(square_image, (int(pt[0]), int(pt[1])), 10, color_dict['cyan'], -1)

        '''
        square image | square candidates image
        inter image  | line image
        '''
        output_image = self.init_resize_image(square_image, 512)
        output_image = np.concatenate([output_image, self.init_resize_image(square_candidate_image, 512)], axis=1)
        output_image_tmp = np.concatenate([self.init_resize_image(inter_image, 512), self.init_resize_image(line_image, 512)], axis=1)
        output_image = np.concatenate([output_image, output_image_tmp], axis=0)

        cv2.imwrite(save_path, output_image[:,:,::-1])

        return output_image


    def save_output(self, session_id, input_image_url, image, output):
        dirpath = os.path.join('static/results', session_id)

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        
        save_path = os.path.join(dirpath, 'output.png')
        self.draw_output(image, output, save_path=save_path)

        output_image_url = os.path.join(dirpath, 'output.png')

        rst = {}
        rst['input_image_url'] = input_image_url
        rst['session_id'] = session_id
        rst['output_image_url'] = output_image_url

        with open(os.path.join(dirpath, 'results.json'), 'w') as f:
            json.dump(rst, f)


def init_worker(args):
    global model
    
    model = model_graph(args)

   
@app.route('/')
def index():
    return render_template('index_scan.html', session_id='dummy_session_id')


@app.route('/', methods=['POST'])
def index_post():
    request_start = time.time()
    configs = request.form
            
    session_id = str(uuid.uuid1())

    image_url = configs['image_url'] # image_url
    
    if len(image_url) == 0:
        bio = BytesIO()
        request.files['image'].save(bio)
        rawimg = bio.getvalue()
        image, image_url = model.decode_image(session_id, rawimg)
    else:
        image = model.read_image(image_url)
    
    output = model.pred_tflite(image)
    
    model.save_output(session_id, image_url, image, output)

    return render_template('index_scan.html', session_id=session_id)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    args = parser.parse_args()

    init_worker(args)

    app.run(host='0.0.0.0', port=args.port)
