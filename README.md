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

<p>This project requires compatible <b>GPU</b> to install tensorflow, you can run it on your local machine in case you have one or use <a href='https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwissLL5-MvxAhXwlYsKHbkBDEUQFnoECAMQAw&url=https%3A%2F%2Fcolab.research.google.com%2Fnotebooks%2F&usg=AOvVaw0eDNVclINNdlOuD-YTYiiB'>Google Colaboratory</a> with <b>Runtime Type<b> changed to GPU</p>
  
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
 
  
</ol>
  

