import numpy as np
import cv2

def get_image_from_request(request):
    filestr = request.files['image'].read()
    nparr = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img