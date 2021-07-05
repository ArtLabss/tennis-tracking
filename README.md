<h1 align='center'>Tennis Tracking</h1>

![Forks](https://img.shields.io/github/forks/ArtLabss/tennis-tracking.svg)
![Stars](https://img.shields.io/github/stars/ArtLabss/tennis-tracking.svg)
![Watchers](https://img.shields.io/github/watchers/ArtLabss/tennis-tracking.svg)
![Last Commit](https://img.shields.io/github/last-commit/ArtLabss/tennis-tracking.svg) 

<h3>Objectvies</h3>
<ul>
  <li>Track the ball </li>
  <li>Detect court lines </li>
  <li>Detect the players</li>
</ul>



<h3>Example using the <a href="VideoInput/video_input1.mp4">sample video</a></h3>

  
Input            |  Output
:-------------------------:|:-------------------------:
![input_img](https://github.com/ArtLabss/tennis-tracking/blob/eb0d21c54467550e52bf77e66091ecff0681605d/input.gif)  |  ![output_img](https://github.com/ArtLabss/tennis-tracking/blob/eb0d21c54467550e52bf77e66091ecff0681605d/output.gif)

<h3>How to run</h3>

<p>This project requires compatible <b>GPU</b> to install tensorflow, you can run it on your local machine in case you have one or use <a href='https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwissLL5-MvxAhXwlYsKHbkBDEUQFnoECAMQAw&url=https%3A%2F%2Fcolab.research.google.com%2Fnotebooks%2F&usg=AOvVaw0eDNVclINNdlOuD-YTYiiB'>Google Colaboratory</a> with <b>Runtime Type<b> changed to GPU.</p>
  
<ol>
  <li>
    Clone this repository
  </li>
  
  ```
  git clone https://github.com/ArtLabss/tennis-tracking
  ```
  
   <li>
     Download yolov3 weights (237 MB) from <a href="https://pjreddie.com/media/files/yolov3.weights">here</a> and add it to your <a href="/Yolov3">Yolov3 directory</a>.
  </li>
  
  <li>
    Install the requirements
  </li>
  
  ```
  pip install -r requirements.txt
  ```
  
  <li>If you are using Google Colab upload all the files to Google Drive</li>
  <li>
    Run predict_video.py (Google colab), you may need to change the folder path/name
  </li>
</ol>
  
  ```
   !python3 predict_video.py  --save_weights_path=WeightsTracknet/model.1 --input_video_path=VideoInput/video_input1t.mp4 --output_video_path=VideoOutput/video_output.mp4 --n_classes=256 --path_yolo_classes=Yolov3/yolov3.txt --path_yolo_weights=Yolov3/yolov3.weights --path_yolo_config=Yolov3/yolov3.cfg
  ```
  
  
<h3>Acknowledgements</h3>
  
<p>To track the ball we used <a href='https://nol.cs.nctu.edu.tw:234/open-source/TrackNet'>TrackNet</a> - deep learning network for tracking high-speed objects
  <br>For court detection line segment and box detection using <a href='https://github.com/navervision/mlsd'>M-LSD</a> was implemented
  <br>And for the players yolov3 was used.
</p>
  
<h3>References</h3>
- Yu-Chuan Huang, "TrackNet: Tennis Ball Tracking from Broadcast Video by Deep Learning Networks," Master Thesis, advised by Tsì-Uí İk and Guan-Hua Huang, National Chiao Tung University, Taiwan, April 2018.
- Yu-Chuan Huang, I-No Liao, Ching-Hsuan Chen, Tsì-Uí İk, and Wen-Chih Peng, "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications," in the IEEE International Workshop of Content-Aware Video Analysis (CAVA 2019) in conjunction with the 16th IEEE International Conference on Advanced Video and Signal-based Surveillance (AVSS 2019), 18-21 September 2019, Taipei, Taiwan.
- 

```
@misc{gu2021realtime,
    title={Towards Real-time and Light-weight Line Segment Detection},
    author={Geonmo Gu and Byungsoo Ko and SeoungHyun Go and Sung-Hyun Lee and Jingeun Lee and Minchul Shin},
    year={2021},
    eprint={2106.00186},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
  ```
</ol>
