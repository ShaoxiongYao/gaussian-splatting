#!/bin/bash
# Frequently used commands

# export QT_QPA_PLATFORM=offscreen
# colmap automatic_reconstructor --workspace_path /home/motion/glasshouse-dataset/pepper_single01\
#                                --image_path /home/motion/glasshouse-dataset/pepper_single01/images\
#                                --quality medium --use_gpu 0

python train.py -s /home/motion/glasshouse-dataset/pepper_single00/

python render.py -m output/904fe9c9-1 --skip_train --skip_test --deform_path /home/motion/visual-tactile-simulate/out_data/sim_orange_tree_leaves_00

# python render.py -m output/e34fe1a5-f --skip_train --skip_test --deform_path /home/motion/visual-tactile-simulate/out_data/sim_fiddle_tree_leaf_03

# python render.py -m output/5a6e7870-8
