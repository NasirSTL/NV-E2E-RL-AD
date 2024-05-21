## Verification

This directory is the implementation of the Verification chapter of the Thesis: **Verifiable End-to-End Reinforcement Learning-Based Autonomous Driving**

*Note:* Currently, this directory does not contain the training scripts for the conditional GANs. If you are interested in the training scripts, please let me know. The following instructions are for training the controller using the bycicle model. 

## Get Started

1. Enter this directory, do not enter the `BC_model_controller` directory:
    ```Shell
    cd verification
    ```

2. To train the controller using the bicycle model:
    ```Shell
    python BC_model_controller/train.py
    ```
    You can modify most of the hyperparameters and other settings in `BC_model_controller/utils/cfg_helper.py` file.

*Note*: this portion of the code is still under development. The current architecture of the controller is probably too simple to be able to do zero-shot transfer. You are welcome and encouraged to modify the controller architecture and hyperparameters to improve the performance of the controller. If you have any questions, feel free to ask.