import torch
import os
import sys
import cv2
sys.path.append(os.path.dirname(__file__))

from models.mbv2_mlsd_tiny import  MobileV2_MLSD_Tiny
from models.mbv2_mlsd_large import  MobileV2_MLSD_Large

from utils import  pred_lines


print(2)
def main():
    current_dir = os.getcwd()
    model_path = current_dir+'/models/mlsd_tiny_512_fp32.pth'
    model = MobileV2_MLSD_Tiny().cuda().eval()

    # model_path = current_dir + '/models/mlsd_large_512_fp32.pth'
    # model = MobileV2_MLSD_Large().cuda().eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    img_fn = current_dir+'/data/lines.png'

    img = cv2.imread(img_fn)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lines = pred_lines(img, model, [512, 512], 0.1, 20)

    for l in lines:
        cv2.line(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (200,0,0), 1,16)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1366, 768))
    cv2.imwrite(current_dir+'/data/frame_1_out.jpg', img)
    
if __name__ == '__main__':
    main()