
## Project 2: Continuous Control


#### Overview of the project
> In this project our goal is to train two agents to control racket to bounce a ball over the net. A reward of 0.1 is provided for every time the agent hits the ball over the net thus enabling the agent to prioritize hitting the ball over the net else it receives a reward of -0.01.. 

> We have 8 possible states corresponding to position and velocity of the ball and racket.Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents)

#### Requirement Gathering And Running Project

> To run this project on your local machine you should have certain packages on your system. 

> The packages required for the project are provided in requirements.yml, use the command conda env update -f requirements.yml to install.

> You can download the required environment using the links below: 

> Windows_32bit : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip

> Windows_64bit : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

> linux : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip

> Mac : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip

> (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

> Place the file in the DRLND GitHub repository, in the p3_collab-compet/ folder, and unzip (or decompress) the file.

> Then run the cells in Tennis.ipynb, then you will be able to train the agent which can solve the task.
