# 2dilc-rl: 2D Iterative Learning Control with Deep Reinforcement Learning Compensation for the Non-repetitive Batch Processes


## Catalog
* env_sys: Two different batch environment--injection molding process and nonlinear batch reactor. 
* ILC_Controller:  Solving the control law for the two-dimensional iterative learning controller.
* DRL_Compensator: Training the two-dimensional DRL compensator.
* Trained_2D_ILC_RL_Controller: the designed and trained 2D ILC-RL controller.
* Supplementary_Material: Supplementary materials for the Journal.
## Getting Started
* Clone this repo: `git clone https://github.com/CrazyThomasLiu/2dilc-rl`
* Create a python virtual environment and activate. `conda create -n 2dilc-rl python=3.8` and `conda activate 2dilc-rl`
* Install dependenices. `cd 2dilc_rl`, `pip install -r requirement.txt` and `cd code; python setup.py develop`

## Calculation of the 2D ICL controller
Run the following command to obtain the control law of the 2D ILC Controller.
```
cd ILC_Controller
python controllaw_injection_molding_process.py or controllaw_nonlinear_batch_reactor.py 
```

## Training of the 2D DRL compensator
```
cd DRL_Compensator
python demo_nominal_injection_molding_process.py or demo_nominal_nonlinear_batch_reactor.py 
python demo_practical_injection_molding_process.py or demo_nominal_nonlinear_batch_reactor.py 
```
The training usually takes 4 hours for the injection_molding_process and 6 hours for the nonlinear batch reactor.


## Test for the control performance simulation 
Run the following command to test the final control performance.
* Injection Molding Process

```
cd Trained_2D_ILC_RL_Controller/Injection_Molding_Process
python demo_injection_molding_process.py
```
![image](https://github.com/CrazyThomasLiu/2dilc-rl/raw/master/Trained_2D_ILC_RL_Controller/Injection_Molding_Process/Injection_molding_output.jpg)


* Nonlinear Batch Reactor

```
cd Trained_2D_ILC_RL_Controller/Nonlinear_Batch_Reactor
python demo_nonlinear_batch_reactor.py
```

![image](https://github.com/CrazyThomasLiu/2dilc-rl/raw/master/Trained_2D_ILC_RL_Controller/Nonlinear_Batch_Reactor/Nonlinear_batch_reactor_output.jpg)





