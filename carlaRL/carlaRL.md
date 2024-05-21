## CARLA-RL

This directory is the implementation of the Reinforcement Learning chapter of the Thesis: **Verifiable End-to-End Reinforcement Learning-Based Autonomous Driving**

## Requirements

* [CARLA 0.9.14](https://github.com/carla-simulator/carla/releases/tag/0.9.14) or later. All of our training and testing was done on CARLA 0.9.14, but everything should work on later versions.

## Get Started

1. Enter the CARLA root directory and launch the CARLA server. Follow the instructions in the [CARLA documentation](https://carla.readthedocs.io/en/latest/) to install and run the CARLA server.

2. On a separate terminal, enter this directory

3. To run the test file:
    ```Shell
    python test.py
    ```
    You will need to modify the `model_path` in `test.py` to the path of the trained model you have.

4. To train a model, we provide a script to train the model using PPO algorithm:
    ```Shell
    python train_ppo.py
    ```
    by default, the training will follow a curriculum learning approach where the agent starts with a simple environment and small number of steps, and gradually increase the complexity of the environment and the number of steps in order to learn better control policies. You can modify the training hyperparameters and some other settings by modifying the arguments in `train_ppo.py`.

## Parameters of the CARLA environment

here we briefly describe the parameters that we have used in the CARLA environment, you can modify them in both the training and testing scripts.

* `host`, `port`, `town`: the host, port, and town of the CARLA server (world).

* `mode`: it can be set to test, train, or train_controller. It controls how the ego vehicle is spawned.

* `algo`: the only relevant value is ppo, which is the algorithm used to train the agent.

* `controller_version`: the relevant values are 1, 2, and 3. 
  * version 1 is most relevant to the thesis, where we explore verification of the lane following controller model.
  * version 3 is most relevant to the physical world of miniature city: the agent is trained on a history of 10 weighted images. This enables the agent to drive in more challenging scenarios such as intersections and roundabouts.

* `model`: the model used for lane detection. The relevant values are openvino, lanenet, and ufld. If you are using the agent model we trained, you should set this to lanenet. 

* `model_path`: the path to the lane detection model. By default, it is set to the LaneNet model we trained. This is the same model used in the f1tenth vehicle. 

* `collect`: this is a boolean value that controls whether to collect data or not. If set to True, the agent will collect data and save it in the `data` directory. This is useful for visualizing and also collecting data for training the generative model or finetuning the agent.

Other parameters should be self-explanatory. The scripts also contain comments that explain the parameters. If you have any questions, feel free to ask.