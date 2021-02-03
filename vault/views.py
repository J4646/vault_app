from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed
import time
import json

#last_request=0.0
#time_limit=10.0


from django.shortcuts import render
def index(request):
  print('**index')
  return render(request,'index.html')


from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import sys
from .ai_classification import ai_check_class
import os

@csrf_exempt
def vault(request):
  if request.method != 'POST' :
    return HttpResponseNotAllowed(['POST'])    
  try:
    data = json.loads(request.body)	
    for i in range(10):
      key='class_'+str(i)  
      img = data[key]
      MODEL=os.environ.get("MODEL", default='secret_model')
      result, tensor =  ai_check_class(img,int(key[-1]), model=MODEL)
      if not result:
        return JsonResponse({'status': "unauthorized/image not recognized","flag":"none","img_class":key, "output_tensor": tensor},status=401)    
  except:
    print('Exception')
    print("Unexpected error:", sys.exc_info()[0])
    raise
    return JsonResponse({'error': "POST data parsing failed"},status=422)
    
  FLAG=os.environ.get("FLAG", default='undefined')
  return JsonResponse({'status': "Authorized","flag":FLAG},status=200)    
  
 
         
 
  
