#!/bin/bash
# Frequently used commands

# colmap automatic_reconstructor --workspace_path /media/motion/IML/mobile_tactile_sensor/test_brown_boot_scan01\
#                                --image_path /media/motion/IML/mobile_tactile_sensor/test_brown_boot_scan01/color --quality medium

# Paths for the COLMAP workspace
database_path="/media/motion/IML/mobile_tactile_sensor/test_tide_jar_touch00/database.db"  # Path to the existing database
image_path="/media/motion/IML/mobile_tactile_sensor/test_tide_jar_touch00/color"           # Path to the directory with new images
model_path="/media/motion/IML/mobile_tactile_sensor/test_tide_jar_touch00/sparse/0"        # Path to the existing sparse model output
output_path="/media/motion/IML/mobile_tactile_sensor/test_tide_jar_touch00/output"        # Path to the existing sparse model output

# Step 1: Feature extraction for the new images
colmap feature_extractor --database_path ${database_path} --image_path ${image_path}

# Step 2: Feature matching for the new images
colmap exhaustive_matcher --database_path ${database_path}

# Step 3: Incremental mapping (incrementally update the existing model)
# colmap mapper --database_path ${database_path} --image_path ${image_path} --output_path ${output_path} --Mapper.num_threads 8
# Step 3: Register the new images to the existing model
colmap image_registrator --database_path ${database_path} --input_path ${model_path} --output_path ${output_path}

# python train.py -s /home/motion/plant-track/data/test_mobile_sensor_01

# python render.py -m output/27069e3c-f --skip_train --skip_test --deform_path /home/motion/visual-tactile-simulate/out_data/sim_orange_tree_leaves_00

# python render.py -m output/688e1100-3 --skip_train --skip_test --deform_path /home/motion/visual-tactile-simulate/out_data/sim_orange_tree_test0

# python render.py -m output/688e1100-3

# for i in {0..46}
# do
# mkdir "/media/motion/IML/mobile_tactile_sensor/test_brown_boot/txt_output/${i}"

# colmap model_converter --input_path "/media/motion/IML/mobile_tactile_sensor/test_brown_boot/sparse/${i}" \
#                        --output_path "/media/motion/IML/mobile_tactile_sensor/test_brown_boot/txt_output/${i}" --output_type TXT
# done

#!/bin/bash

# # Define the base path for the models
# base_path="/media/motion/IML/mobile_tactile_sensor/test_brown_boot/sparse"
# output_path="/media/motion/IML/mobile_tactile_sensor/test_brown_boot/merged_model"  # Final output path

# # Create the initial merged directory if it doesn't exist
# mkdir -p "$output_path"

# # Initialize the merge process with the first model
# merged_model="${base_path}/0"

# # Loop through each model and merge it with the current merged model
# for i in $(seq 1 2); do
#     echo "current merged model: ${merged_model}"
#     next_model="${base_path}/${i}"
#     echo "next model: ${next_model}"
#     temp_output="${base_path}/temp_merged_model_$i"

#     # Merge the current merged model with the next model
#     colmap model_merger \
#         --input_path1 "$merged_model" \
#         --input_path2 "$next_model" \
#         --output_path "$temp_output"

#     # Update the merged model path to the latest output
#     merged_model="$temp_output"
# done

# # Move the final merged model to the desired output path
# mv "$merged_model" "$output_path"

# # Cleanup temporary files if needed
# rm -rf "${base_path}/temp_merged_model_"