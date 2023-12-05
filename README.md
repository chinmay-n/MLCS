# This is a modified readme from the github repo pointed to in the assignment

```bash
├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
    └── anonymous_1_poisoned_data.h5
    └── Multi-trigger Multi-target
        └── eyebrows_poisoned_data.h5
        └── lipstick_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
├── models
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5
    └── anonymous_1_bd_net.h5
    └── anonymous_1_bd_weights.h5
    └── anonymous_2_bd_net.h5
    └── anonymous_2_bd_weights.h5
    └── x2_sunglasses_bd_net.h5
    └── x4_sunglasses_bd_net.h5
    └── x10_sunglasses_bd_net.h5
├── architecture.py
└── eval.py // this is the evaluation script
```

## I. Validation Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
   3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5. Similarly, there are other .h5 files with poisoned data that correspond to different BadNets under models directory.

## II. Evaluating the Backdoored Model
   There have been mior changes to the eval.py script to accomodate the new model structure. These are outlined below:

   1. The DNN architecture used to train the face recognition model is the state-of-the-art DeepID network. This DNN is backdoored with multiple triggers. Each trigger is associated with its own target label. 
   
   2. To generate the repaired models, execute `repair.py`by running:  
      `python eval.py <clean validation data directory> <model directory>`.
      
      E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`. Clean data classification accuracy on the provided validation dataset for sunglasses_bd_net.h5 is 97.87 %.
   
   3. To evaluate the backdoored model, execute `eval.py` by running:  
      `python eval.py <clean validation data directory> <base model directory> <modified model directory>`.
      
      E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5 models/x2_sunglasses_bd_net.h5`.
