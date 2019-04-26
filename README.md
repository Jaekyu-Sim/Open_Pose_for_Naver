# Open_Pose_for_Naver

I select <OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (https://arxiv.org/abs/1812.08008)> for evaluating my code.

The Loss of this project is expressed here.

![Confidencemap_Loss](/readme_image/Heatmap_loss.png)
![PAFs_output1](/readme_image/Vectormap_loss.png)

the left is heatmap loss and right is vectormap loss.

In my project, there are many python files.

1. For_parsing_mat_to_json.ipynb
this ipynb file makes .mat file to json file. When you download 'mpii_human_pose_dataset', there are annotation data. And the annotation data's format is .mat. So you convert this file format to json.

2. Image_resize.ipynb
this ipynb file makes image files to regular size(regular width x regular height). The reason why I make image files size regulary is for saving time. By making image file size regular, training time could be reduced. Because there are no time to resize image during training.

3. network.py and network.ipynb
this files are containing VGG-19 network.

4. Pose.ipynb
this ipynb file is for training. It contains class and util function to train.

5. Demo.ipynb(not important)
this ipynb file is for making skeleton map(reference. https://github.com/evalsocket/tf-openpose)


Here is a output of this project. There are vectormap and heatmap images and output of my network.

input images
![input_image](/readme_image/input_image.png)

1. Heatmap images


![Heatmap_output1](/readme_image/heatmap.png)



2. Vectormap images
![Heatmap_output2](/readme_image/paf.png)

3. Heatmap of right knee images
![Vectormap_output1](/readme_image/heatmap_right_knee.png)

4. paf of right arm images
![Vectormap_output2](/readme_image/paf_right_arm.png)

5. cv2.circle images of non-maxima and 
![Heatmap_output3](/readme_image/heatmap_all_peaks.png)

6. otuput images of my Network!!!! 
![output1](/readme_image/output.png)

