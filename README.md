实验目的：pytorch 联合身体和脸部进行性别判断。

配置：   python3.6.5
	 
	 pytorch

实验流程：
1.
  下载实验图片。有两个实验图片文件夹。一个名为body，内含人物的上半身图片。一个名为face,内含body里面人物的脸部图片。
	body 下载地址：https://pan.baidu.com/s/1SakkoRiu9x0vha_DewR6GA  。
	face 下载地址：https://pan.baidu.com/s/1fd3I3yZWTzVnhXWmi6uD9w  。
    下载解压后，将两个文件夹放到gender_recognize.py 同一目录下。
    
2.下载权值文件。需下载两个权值文件。一个名为ckpt_body.pth ,下载地址： https://pan.baidu.com/s/1l40oOJkx0ka--cWBy4sILA  。
   一个名为ckpt_face.pth,下载地址： https://pan.baidu.com/s/15Vni4Y9E_kzQv878qsYEEg  。下载之后，都放入models文件夹下。
   
3.运行gender_recognize.py.在输出可以看到叛断错误的图片名，以及最后的判断准确率。
 

	
