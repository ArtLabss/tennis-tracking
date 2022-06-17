<p align='center'>
  <a href="https://www.artlabs.tech"><img src='https://raw.githubusercontent.com/ArtLabss/tennis-tracking/main/VideoOutput/artlabs%20logo.jpg' width="150" height="170"></a>
</p>

<h1 align='center'>Tennis Tracking 🎾</h1>
<p align='center'>
  <img src="https://img.shields.io/github/forks/ArtLabss/tennis-tracking.svg">
  <img src="https://img.shields.io/github/stars/ArtLabss/tennis-tracking.svg">
  <img src="https://img.shields.io/github/watchers/ArtLabss/tennis-tracking.svg">
  
  <br>
  
  <img src="https://img.shields.io/github/last-commit/ArtLabss/tennis-tracking.svg">
  <img src="https://img.shields.io/badge/license-Unlicense-blue.svg">
  <img src="https://hits.sh/github.com/ArtLabss/tennis-tracking.svg"/>
  <br>
  <code>With ❤️ by ArtLabs</code>
  
</p>

<!-- 
![Forks](https://img.shields.io/github/forks/ArtLabss/tennis-tracking.svg)
![Stars](https://img.shields.io/github/stars/ArtLabss/tennis-tracking.svg)
![Watchers](https://img.shields.io/github/watchers/ArtLabss/tennis-tracking.svg)
![Last Commit](https://img.shields.io/github/last-commit/ArtLabss/tennis-tracking.svg)  
-->

<h3>Objectives</h3>
<ul>
  <li>Track the ball </li>
  <li>Detect court lines </li>
  <li>Detect the players</li>
</ul>

<p>To track the ball we used <a href='https://nol.cs.nctu.edu.tw:234/open-source/TrackNet'>TrackNet</a> - deep learning network for tracking high-speed objects. For players detection ResNet50 was used. See <a href="https://artlabs.tech/projects/"> ArtLabs/projects</a> for more or similar projects.</p>


<h3>Example using <a href="https://github.com/ArtLabss/tennis-tracking/tree/main/VideoInput">sample videos</a></h3>

  
Input            |  Output
:-------------------------:|:-------------------------:
![input_img1](https://github.com/ArtLabss/tennis-tracking/blob/00cfe10b18db1e6a68800921dfbda010f90a74bb/VideoOutput/ezgif.com-gif-maker(3).gif)  |  ![output_img1](https://github.com/ArtLabss/tennis-tracking/blob/0f684fdeef96a715984dc74b62b961f68ff95edc/VideoOutput/ezgif.com-gif-maker.gif)
![input_img2](https://github.com/ArtLabss/tennis-tracking/blob/579fb3344935bbf4c5d08e27c99ffc6b56bed896/VideoOutput/ezgif.com-gif-maker(1).gif)  |  ![output_img2](https://github.com/ArtLabss/tennis-tracking/blob/579fb3344935bbf4c5d08e27c99ffc6b56bed896/VideoOutput/ezgif.com-gif-maker(2).gif)
![input_img3](https://github.com/ArtLabss/tennis-tracking/blob/06179bdd29d4424f5e19e5600802f853aaa86f22/VideoOutput/monteCarlo_input.gif)  |  ![output_img3](https://github.com/ArtLabss/tennis-tracking/blob/06179bdd29d4424f5e19e5600802f853aaa86f22/VideoOutput/monteCarlo_output.gif)

<h3>How to run</h3>

<p>This project requires compatible <b>GPU</b> to install tensorflow, you can run it on your local machine in case you have one or use <a href='https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwissLL5-MvxAhXwlYsKHbkBDEUQFnoECAMQAw&url=https%3A%2F%2Fcolab.research.google.com%2Fnotebooks%2F&usg=AOvVaw0eDNVclINNdlOuD-YTYiiB'>Google Colaboratory</a> with <b>Runtime Type</b> changed to <b>GPU</b>.</p>

- [ ] Input videos have to be rallies of the game and shouldn't contain any <strong>commercials, breaks or spectators</strong>.
  
<ol>
  <li>
    Clone this repository
  </li>
  
  ```git
  git clone https://github.com/ArtLabss/tennis-tracking.git
  ```
  
   <li>
     Download yolov3 weights (237 MB) from <a href="https://pjreddie.com/media/files/yolov3.weights">here</a> and add it to your <a href="/Yolov3">Yolov3 folder</a>.
  </li>
  
  <li>
    Install the requirements using pip 
  </li>
  
  ```python
  pip install -r requirements.txt
  ```
  
   <li>
    Run the following command in the command line
  </li>
  
  ```python
  python3 predict_video.py --input_video_path=VideoInput/video_input3.mp4 --output_video_path=VideoOutput/video_output.mp4 --minimap=0 --bounce=0
  ```
  
  <li>If you are using Google Colab upload all the files to Google Drive, including yolov3 weights from step <strong>2.</strong></li>
  
   <li>
    Create a Google Colaboratory Notebook in the same directory as <code>predict_video.py</code>, change Runtime Type to <strong>GPU</strong> and connect it to Google drive
  </li>
  
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
  
  <li>
    Change the working directory to the one where the Colab Notebook and <code>predict_video.py</code> are. In my case,
  </li>
  
  ```python
  import os 
  os.chdir('drive/MyDrive/Colab Notebooks/tennis-tracking')
  ```
  
  <li>
    Install only 2 requirements, because Colab already has the rest
  </li>
  
  ```python
  !pip install filterpy sktime
  ```
  
  <li>
    Inside the notebook run <code>predict_video.py</code>
  </li>
  
  ```
   !python3 predict_video.py --input_video_path=VideoInput/video_input3.mp4 --output_video_path=VideoOutput/video_output.mp4 --minimap=0 --bounce=0
  ```
  
  <p>After the compilation is completed, a new video will be created in <a href="/VideoOutput" target="_blank">VideoOutput folder</a> if <code>--minimap</code> was set <code>0</code>, if <code>--minimap=1</code> three videos will be created: video of the game, video of minimap and a combined video of both</p>
  <p><i>P.S. If you stumble upon an <b>error</b> or have any questions feel free to open a new <a href='https://github.com/ArtLabss/tennis-tracking/issues'>Issue</a> </i></p>
  
</ol>


<h3>What's new?</h3>
<ul>
  <li>Court line detection improved</li>
  <li>Player detection improved</li>
  <li>The algorithm now works practically with any court colors</li>
  <li>Faster algorithm</li>
  <li>Dynamic Mini-Map with players and ball added, to activate use argument <code>--minimap</code></li>
  </ul>
  
`--minimap=0`            |  `--minimap=1`
:-------------------------:|:-------------------------:
![input_img1](https://github.com/ArtLabss/tennis-tracking/blob/4b5ff2849b71af67023c4160c4f91481a6821bb3/VideoOutput/input6.gif)  |  ![output_img1](https://github.com/ArtLabss/tennis-tracking/blob/3124a8609b30deb557c1563c45febb1fd86c8956/VideoOutput/input3.gif)

<p>
  To predict bounce points machine learning library for time series <a href="https://www.sktime.org/en/stable/index.html">sktime</a> was used. Specifically, <a href="https://github.com/ArtLabss/tennis-tracking/blob/90652b4547311423ea49c4195dde9da9a81f1893/clf.pkl">TimeSeriesForestClassifier</a> was trained on 3 variables:  <code>x</code>, <code>y</code> coordinates of the ball and <code>V</code> for velocity (<code>V2-V1/t2-t1</code>). Data for training the model - <a href="https://github.com/ArtLabss/tennis-tracking/blob/main/bigDF.csv" >df.csv</a>
<p>
<ul>
  <li>By specifiying <code>--bounce=1</code> bounce points can be detected and displayed</li>
</ul>
<p align="center">
  <kbd>
  <img width=500 src="https://github.com/ArtLabss/tennis-tracking/blob/a6f395716dc5a076bfb2fc49f97db96a2004efed/VideoOutput/9bounces.gif">
  </kbd>
</p>

<p>
  The model predicts true negatives (not bounce) with accuracy of <strong>98%</strong> and true positives (bounce) with <strong>83%</strong>.
</p>


<h3>Further Developments</h3>
<ul>
  <li><strike>Improve line detection of the court and remove overlapping lines</strike></li>
  <li><strike>Algorithm fails to detect players when the court colors aren't similar to the sample video</strike></li>
  <li><strike>Don't detect the ballboys/ballgirls</strike></li>
  <li><strike>Don't contour the banners</strike></li>
  <li><strike>Find the coordinates of the ball touching the court and display them</strike></li>
  <li>Code Optimization</li>
  <li><strike>Dynamic court mini-map with players and the ball</strike></li>
</ul>

<h3>Current Drawbacks</h3>
<ul>
  <li>Slow algorithms (to process 15 seconds video (6.1 Mb) it takes <strike>28 minutes</strike> 16 minutes)<br><ul><li>Instead of writing a new video, a faster way would be to show the frame right after it has been processed</li></ul></li>
  <li>Algorithm works only on official match videos</li>
</ul>
 
<h3>Helpful Repositories</h3>
<ul>
  <li><a href="https://github.com/MaximeBataille/tennis_tracking">Tennis Tracking</a> @MaximeBataille</li>
  <li><a href="https://github.com/avivcaspi/TennisProject">Tennis Project</a> @avivcaspi</li>
  <li><a href="https://nol.cs.nctu.edu.tw:234/open-source/TrackNet/tree/master/Code_Python3">TrackNet</a></li>
</ul>

<h3>Contribution</h3>

<p>Help us by contributing, check out the <a href="https://github.com/ArtLabss/tennis-tracking/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a>. Contributing is easy!</p>

<h3>References</h3>

- Yu-Chuan Huang, "TrackNet: Tennis Ball Tracking from Broadcast Video by Deep Learning Networks," Master Thesis, advised by Tsì-Uí İk and Guan-Hua Huang, National Chiao Tung University, Taiwan, April 2018. 

- Yu-Chuan Huang, I-No Liao, Ching-Hsuan Chen, Tsì-Uí İk, and Wen-Chih Peng, "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications," in the IEEE International Workshop of Content-Aware Video Analysis (CAVA 2019) in conjunction with the 16th IEEE International Conference on Advanced Video and Signal-based Surveillance (AVSS 2019), 18-21 September 2019, Taipei, Taiwan.

- Joseph Redmon, Ali Farhadi, "YOLOv3: An Incremental Improvement", University of Washington, https://arxiv.org/pdf/1804.02767.pdf
