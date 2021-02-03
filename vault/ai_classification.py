import cv2
import numpy as np
import base64
import json

def ai_check_class(img, expected_class, confidence=0.8, model='mnist_model'):
  img = data_uri_to_cv2_img(img)
  img = img_to_tf_serving_json(img)
  prediction = tf_api_predict(img,model)
  prediction = prediction['predictions'][0]
  return evaluate_prediction(prediction,expected_class, confidence)

def evaluate_prediction(model_pred, expected_class, confidence=0.8):
  match=False
  print('pred', model_pred)
  print('class',expected_class)
  
  if np.argmax(model_pred)==expected_class:
    if model_pred[expected_class]>=confidence:
      match=True  
  return match,model_pred

def data_uri_to_cv2_img(uri):
#convert data uri -'data:image/png;base64,iVBOR......' to openCV object
  encoded_data = uri.split(',')[1]
  encoded_data=encoded_data.encode('ascii')
  decoded_data = base64.b64decode(encoded_data)
  nparr = np.fromstring(decoded_data, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
  return img

def img_to_tf_serving_json(img):
#convert openCV img object to json for tensorflow serving REST API 
#model expect input shape (1,28,28,1) (normalized)
  img=img/255 #normalize
  img=img.reshape(1,28,28,1)
  img_str = np.array_repr(img).replace('\n','').replace(' ','')
  img_str = img_str.replace(')','')
  img_str = img_str.replace('(','')
  img_str = img_str.replace('array','')
  img_str = img_str.replace('.]','.0]')  # to avoid parsing err JSON Parse error: Miss fraction part in number.
  img_json='{"instances":'+img_str+'}'
  return img_json 

from urllib import request   
def tf_api_predict(img, model='mnist_model'):
#img - opencv img object
#call tf serving REST API with image json data
#interface with tensorflow serving @localhost:8501
#returns output tensor as array
#generic api call:
#curl -d @answer_5.dat -X POST http://localhost:8501/v1/models/mnist_model:predict  
#curl -d {"test":"test"} -X POST http://tf_serving:8501/v1/models/mnist_model:predict  
  url='http://tf_serving:8501/v1/models/{}:predict'.format(model)  
  req = request.Request(url, data=str.encode(img))
  r = request.urlopen(req)
  if r.status == 200:
    dd = r.read().decode('utf-8')
    j = json.loads(dd)
    print('TF result',j)
    return(j)
  else:
    print('Error openning url:', url)
  return False 
