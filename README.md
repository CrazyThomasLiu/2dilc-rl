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
* Injection Molding Process

```
cd Trained_2D_ILC_RL_Controller/Injection_Molding_Process
python demo_injection_molding_process.py
```
<img src="Trained_2D_ILC_RL_Controller/Injection_Molding_Process/Injection_molding_output.pdf" width="800" height="223"/> 

![Image text](https://github.com/CrazyThomasLiu/2dilc-rl/blob/master/Trained_2D_ILC_RL_Controller/Injection_Molding_Process/Injection_molding_output.pdf)


## Acknowledgement
We have used codes from other great research work, including [VolSDF](https://github.com/lioryariv/volsdf), [NeRF++](https://github.com/Kai-46/nerfplusplus), [SMPL-X](https://github.com/vchoutas/smplx), [Anim-NeRF](https://github.com/JanaldoChen/Anim-NeRF), [I M Avatar](https://github.com/zhengyuf/IMavatar) and [SNARF](https://github.com/xuchen-ethz/snarf). We sincerely thank the authors for their awesome work! We also thank the authors of [ICON](https://github.com/YuliangXiu/ICON) and [SelfRecon](https://github.com/jby1993/SelfReconCode) for discussing experiment.

## Related Works 
Here are more recent related human body reconstruction projects from our team:
* [Jiang and Chen et. al. - InstantAvatar: Learning Avatars from Monocular Video in 60 Seconds](https://github.com/tijiang13/InstantAvatar)
* [Shen and Guo et. al. - X-Avatar: Expressive Human Avatars](https://skype-line.github.io/projects/X-Avatar/)
* [Yin et. al. - Hi4D: 4D Instance Segmentation of Close Human Interaction](https://yifeiyin04.github.io/Hi4D/)

```
@inproceedings{guo2023vid2avatar,
      title={Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via Self-supervised Scene Decomposition},
      author={Guo, Chen and Jiang, Tianjian and Chen, Xu and Song, Jie and Hilliges, Otmar},    
      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
      year      = {2023}
    }
```

