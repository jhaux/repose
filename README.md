Repose Evaluation pipeline
==========================

__To be used with the edflow package.__

This package supplies code to evaluate the performance of the pose recreation of generative models.

Installation
------------

```
git clone https://github.com/jhaux/repose.git
cd repose
pip install -e .
```

Usage
-----
__Take a look at the documentation of the eval pipeline__ (Todo: link)
After installation you now have access to the callback `repose.repose.default_repose_eval`. (Todo: Nameing should be redone.)
When running the callback you can specify the following keys in your edflow config under the key ``repose_kwargs``:
```
# config.yaml
repose_kwargs:
  data_out_im_key: frame_gen
  data_in_im_key: 'target'                                    
  data_in_kp_key: 'keypoints'                                 
  data_out_im_key: 'frame_gen'                                
  data_out_kp_key: 'keypoints'                               
  metrics: ['l2', 'pck']                                    
  metrics_kwargs: {'l2': {}, 'pck': {'threshold': PCK_THRESH}}
  backend: 'openpose'                                   
  force_recalculation: False                                  
  strategy: 'calc_all'                                        
  backend_kwargs: {}                                         
  num_pose_render: 500                                       

```
Parameters:                                                                   
-----------                                                                   
 - `data_in_im_key` : `str`                                                          
     Key in labels of the ``data_in`` dataset, at which the                    
     ground truth image can be found.                                          
 - `data_in_kp_key` : `str`                                                          
     Key in labels of the ``data_in`` dataset, at which the ground truth       
     keypoints can be found.                                                   
 - `data_out_im_key` : `str`                                                         
     Key in labels of the ``data_out`` dataset, at which the                   
     generated image can be found.                                             
 - `data_out_kp_key` : `str`                                                         
     Key in labels of the ``data_out`` dataset, at which the keypoints         
     estimated from the generated image can be found. If this key is           
     found in the labels, no re-estimation of keypoints on the generated       
     images is done. If it is not found, the keypoints are estimated.          
 - `metrics` : `list(str)`                                                           
     Defines the way the keypoints are compared. Must be one of                
         - ``l2``                                                              
         - ``pck``                                                             
 - `metrics_kwargs` : `dict(str, dict)`                                              
     Keyword Arguments passed to the metric functions each time they are       
     called. If metrics is ``['l2']`` metrics_kwargs must be ``{'l2': {...}}``.
 - `backend` : `str`                                                                 
     Defines the keypoint estimator. Must be one of                            
         - ``openpose``                                                        
         - ``alphapose`` (not yet implemented)                                 
 - `force_recalculation` : `bool`                                                    
     If set to True, will re-estimate the keypoints on the generated           
     images.                                                                   
 - `strategy` : `str`                                                                
     What to do if the keypoint models of the backend and the ground           
     truth keypoints do not match, i.e. one is openpose BODY_25 and the        
     other is COCO_17.                                                         
         - ``calc_all``: Will also estimate the keypoints of the ground        
             truth images if model mismatch is detected. This will add         
             a key to ``data_out.labels`` of the form                          
             ``keypoints.model``, which is checked the next time this          
             callback is run on the data. If ``force_recalculation`` is        
             ``False`` at that point, these keypoints are loaded and           
             used, so that no recalculation is needed.                         
         - ``raise``: Will raise an error if model mismatch is detected.       
 - `backendkwargs` : `dict`                                                          
     Keyword arguments passed to the backend at construction time.             
 - `num_pose_render` : `int`                                                         
     The number of images for which the pose                                   
     detecionts are rendered on top of the frames. Will be turned              
     into a video afterwards using ffmpeg.                                     
