"""
Midtrain the model. Same as pretraining but simpler.
Run as:

python -m scripts.mid_train

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
"""

