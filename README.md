# Open_Pose_for_Naver

I select <OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (https://arxiv.org/abs/1812.08008)> for evaluating my code.

The Loss of this project is expressed here.

![Confidencemap_Loss](/output_for_github/Heatmap_loss.png)
![PAFs_output1](/output_for_github/Vectormap_loss.png)

the left is heatmap loss and right is vectormap loss.

In my project, there are many python files.

1. For_parsing_mat_to_json.ipynb
this ipynb file makes .mat file to json file. When you download 'mpii_human_pose_dataset', there are annotation data. And the annotation data's format is .mat. So you convert this file format to json.

2. Image_resize.ipynb
this ipynb file makes image files to regular size(regular width x regular height). The reason why I make image files size regulary is for saving time. By making image file size regular, training time could be reduced. Because there are no time to resize image during training.

3. network.py
this files are containing VGG-19 network and sub networks.

4. Open_Pose.ipynb
this ipynb file is for training. It contains class and util function to train.

5. Demo.ipynb
this ipynb file is for making skeleton map(reference. https://github.com/evalsocket/tf-openpose)


Here is a output of this project. There are vectormap and heatmap images and output of my network.

input images

![input_image](/output_for_github/input_image.png)

1. Heatmap images


![Confidencemap_output](/output_for_github/output_heatmap.png)


2. Vectormap images

![PAFs_output](/output_for_github/output_vectormap.png)


3. otuput images of my Network

![output](/output_for_github/output.png)

