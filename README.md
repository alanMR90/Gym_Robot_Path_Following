# OpenAI Gym Custom Environment
## Manipulator Robot Path Following 
In this repository you will find all the python codes to recreate the experiments for the paper **A Visual Path-following Learning Approach for Industrial Robots using DRL**.  

To recreate the experiments it is important to have access to a computer with GPU capable of running Tensorflow with CUDA.

> **Note:** The GPU and CUDA configuration are out of the scope of the installation instructions explained here
>
After installation, 3 new gym environments will be registered:
- *Tool5D-v0*: Full-state observation environment using the Tool Center Point 5 DoF (X, Y, Z, Yaw, Pitch) error.

- *Laser2DPoint-v0*: Partial-state observation environment using an RGB image as the observation using a circular red dot as a reference simulating a real laser. On this environment the agent will perform only 2D path following (X, Y) due to the shape of the laser.

- *Laser2DLine-v0*: Partial-state observation environment using an RGB image as the observation using a red line as a reference simulating a real laser. On this environment the agent will perform 3D path following (X, Y, Roll) due to the shape of the laser.

## Installation
1.- Clone this repository
2.- On line 32 of *Gym_Robot_Path_Following/path_following/envs/envUtils/utils.py*
change the path to the corresponding path on your PC. More JPG texture images can be added to this path to have more variety during the training.
```bat
textures = glob.glob("/home/alan/Documents/Github/Gym_Robot_Path_Following/path_following/envs/textures/*.jpg,recursive=True)
```

3.- Run in terminal 


```bat
pip install -e gym_robot_path_following
```
*setup.py* will try to install the most important packages needed to run the custom gym environment.

> **Note:** It is recommended to use Anaconda or another virtual environment package manager.
