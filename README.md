## Verifiable End-to-End Lane Following with Reinforcement Learning

Implementation of RL and Verification parts of the Thesis: **Verifiable End-to-End Lane Following with Reinforcement Learning**<br>
Xinhang (Owen) Ma<br>

For codes and documentations of the other parts of the project, please refer to [f1tenth documentation](/assets/F1Tenth_Documentation.pdf) or contact us by email.

## Requirements

* Linux and Windows, but we recommend Linux for performance and compatibility reasons. All our training and testing was done on Ubuntu 20.04.
* GPUs with enough memory: most RL algorithms require at least 8-10 GB of GPU memory (although some can work with less). Verification framework requires somewhere around the same amount of memory. Training the Generative model requires at least 12-16 GB of GPU memory.
* Python 3.7 and PyTorch 1.3.1
* CUDA 11.0 or later.
* Python libraries: see [requirements.txt](requirements.txt) for a list of libraries that need to be installed. We recommend using a virtual environment to avoid conflicts.

## Installation

1. Clone this repository:
    ```Shell
    git clone 
    cd 
    ```

2. Install the required libraries (we use Conda as an example):
    ```Shell
    conda create -n lane_following python=3.7 -y
    conda activate lane_following
    ```

3. Install the required libraries:
    ```Shell
    pip install -r requirements.txt
    ```

## Getting Started

We provide two different trained RL models for lane following, they are both trained using PPO algorithm. You can find them in the `verification/models` folder.

For the Generator models, you can download trained models from: [Google Drive](https://drive.google.com/drive/folders/1PZSltWbhaq7YG0EX3oVQiOhthMaYsQN0?usp=sharing).
