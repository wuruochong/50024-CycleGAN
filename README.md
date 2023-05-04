# 50024-CycleGAN
50024ML Project: CycleGAN Re-implementation

To train:
run 'python train.py'
By default this trains on the horse2zebra dataset. To train on a different dataset (e.g. monet2photo), modify the main function in train.py to pass in the correct dataset name.

To test:
run 'python test.py'
By default this tests on the horse2zebra dataset. To test on a different dataset (e.g. monet2photo), modify the main function in test.py to pass in the correct dataset name.
Testing loads training checkpoints, so at least one training checkpoint must exist before testing. By default a checkpoint is created every 20 epochs.