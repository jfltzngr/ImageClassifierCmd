# ImageClassifierCmd
Example Code

Dear WandB-Team, I try to implement Hyperparametertuning for my image classification model. I've already finnished a jupyter notebook generating sweeps with the sweep agent. When I try to do the same inside a Python script which is callable together with an argumentparser for data_file, directory to my checkpoints, etc... 
I get this exception: \lib\site-packages\wandb\sdk\wandb_init.py", line 798, in init
six.raise_from(Exception("problem"), error_seen)
File "<string>", line 3, in raise_from
Exception: problem. I couldn't find an example project facing my issue here. (I'm using Pytorch by the way). It would be great if you have an idea how i could tackle this problem. Thanks in advance for your commitment! Best regards Johannes
