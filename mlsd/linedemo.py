from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from utils import pred_lines, pred_squares
import gradio as gr
from urllib.request import urlretrieve


# Load MLSD 512 Large FP32 tflite
model_name = 'tflite_models/M-LSD_512_large_fp32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_name)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def gradio_wrapper_for_LSD(img_input, score_thr, dist_thr):
  lines = pred_lines(img_input, interpreter, input_details, output_details, input_shape=[512, 512], score_thr=score_thr, dist_thr=dist_thr)
  img_output = img_input.copy()

  # draw lines
  for line in lines:
    x_start, y_start, x_end, y_end = [int(val) for val in line]
    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [0,255,255], 2)
  
  return img_output

urlretrieve("https://www.digsdigs.com/photos/2015/05/a-bold-minimalist-living-room-with-dark-stained-wood-geometric-touches-a-sectional-sofa-and-built-in-lights-for-a-futuristic-feel.jpg","example1.jpg")
urlretrieve("https://specials-images.forbesimg.com/imageserve/5dfe2e6925ab5d0007cefda5/960x0.jpg","example2.jpg")
urlretrieve("https://images.livspace-cdn.com/w:768/h:651/plain/https://jumanji.livspace-cdn.com/magazine/wp-content/uploads/2015/11/27170345/atr-1-a-e1577187047515.jpeg","example3.jpg")
sample_images = [["example1.jpg", 0.2, 10.0], ["example2.jpg", 0.2, 10.0], ["example3.jpg", 0.2, 10.0]]



iface = gr.Interface(gradio_wrapper_for_LSD,
                     ["image",
                      gr.inputs.Number(default=0.2, label='score_thr (0.0 ~ 1.0)'),
                      gr.inputs.Number(default=10.0, label='dist_thr (0.0 ~ 20.0)')
                     ],
                     "image",
                     title="Line segment detection with Mobile LSD (M-LSD)",
                     description="M-LSD is a light-weight and real-time deep line segment detector, which can run on GPU, CPU, and even on Mobile devices. Try it by uploading an image or clicking on an example. Read more at the links below",
                     article="<p style='text-align: center'><a href='https://arxiv.org/abs/2106.00186'>Towards Real-time and Light-weight Line Segment Detection</a> | <a href='https://github.com/navervision/mlsd'>Github Repo</a></p>",
                     examples=sample_images,
                     allow_screenshot=True)

iface.launch()