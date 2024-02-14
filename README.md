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

Running this project requires a compatible GPU with tensorflow installed, this can be done on your local machine or on [Google Colab](https://colab.research.google.com/notebooks) by following the steps below.

<a target="_blank" href="https://colab.research.google.com/github/ArtLabss/tennis-tracking/blob/main/Tennis_Tracking_Colab_Setup.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**1. Set Up Your Environment**
- Ensure your Google Colab session is set to use a GPU. You can do this by going to `Runtime` > `Change runtime type` and selecting one of the GPU options as the hardware accelerator.

**2. Clone the Repository**
- Clone the Tennis Tracking repository directly into your Colab environment.
```python
!git clone https://github.com/ArtLabss/tennis-tracking.git
```

**3. YOLOv3 Weights Download**
- Use the following command to download the YOLOv3 weights directly into the Yolov3 folder within your cloned repository.
```python
!wget https://pjreddie.com/media/files/yolov3.weights -P tennis-tracking/Yolov3/
```

**4. Install Dependencies**
- Install the required libraries. Google Colab already includes many dependencies, so you may only need to install a few missing ones.
```python
!pip install -r tennis-tracking/requirements.txt filterpy sktime
```
- For any specific dependencies not included in the requirements file but necessary for Colab, install them using `!pip install` commands.

**5. Prepare Your Input Video**
- Upload your video to Google Drive or directly to Colab using the sidebar. If using Google Drive, mount your drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
- Make sure your input video follows the guidelines (e.g., contains only game rallies, no commerical breaks).

**6. Running the Script**
- Change the directory to the cloned repository's root.
```python
import os
os.chdir('/content/tennis-tracking')
```
- Execute the script with the necessary arguments. Here, replace `path_to_your_video` with the actual path to your video in Colab or Google Drive.
```python
!python predict_video.py --input_video_path=path_to_your_video --output_video_path=VideoOutput/video_output.mp4 --minimap=0 --bounce=0
```

### Additional Tips
- **Automating Video Uploads:** Use the Colab file upload utility to upload videos directly to your Colab session.
```python
from google.colab import files
uploaded = files.upload()
# Follow the prompt to select and upload your video file.
```

- **Accessing Output Videos:** Output videos are saved in the specified `VideoOutput` directory within the Colab environment. You can download them to your local machine using:
```python
from google.colab import files
files.download('VideoOutput/video_output.mp4')
# Replace 'VideoOutput/video_output.mp4' with the correct path if needed.
```

- **Troubleshooting:** If you encounter errors or issues, check the console output for any messages indicating missing dependencies or errors in the script. You can also refer to the [project's issues page](https://github.com/ArtLabss/tennis-tracking/issues) for solutions or to report new issues.

By following these streamlined steps, you can set up and run the Tennis Tracking project on Google Colab.


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
