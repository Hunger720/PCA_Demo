#include <opencv2/opencv.hpp>
#include <iostream>
#include<stdlib.h>

using namespace cv;
using namespace std;

void main(){

	int nsamples;        //样本（图片）个数
	int w = 92;          //图片宽度（假定所有样本已知且统一）
	int h = 112;         //图片高度（假定所有样本已知且统一）
	int index = 0;       //样本序号，从0开始

	char filename[150];  //样本文件路径

	Mat img, gray;
	Mat inputs;          //样本数据矩阵
	Mat result;

	PCA pca;

	//1.输入样本数据
	cin>>nsamples;
	inputs.create(nsamples,h*w,CV_8UC1);

	for(int i=0;i<nsamples;i++){

		cin>>filename;
		img = imread(filename);
		cvtColor(img,gray,CV_BGR2GRAY);

		//将一个h*w的数据样本转换成1*n的向量，n=h*w，即每个样本为一个n维向量
		//将转换成n维向量的样本逐个添加到样本矩阵中，矩阵的每一行为一个样本
		for(int j= 0;j<h;j++)
			for(int k=0;k<w;k++){
				inputs.at<uchar>(index,j*w+k) = gray.at<uchar>(j,k);
			}
		index++;
	}

	//2.进行PCA的相关计算

	//opencv的PCA类输入矩阵的数据类型要求为float型，既CV_32FC1或CV_64FC1
	inputs.convertTo(inputs,CV_32FC1);

	//最后的参数10，保留前10个最大的特征值及其对应的特征向量
	pca(inputs,Mat(),CV_PCA_DATA_AS_ROW,10);

	//读取协方差矩阵的特征向量，并转换成样本（图片）的原尺寸h*w
	//row(0)表示读取第一个（特征值最大）的特征向量
	result = pca.eigenvectors.row(0).reshape(1,h);

	//将向量元素值的范围从浮点值映射到0~255，并将数据类型转换成unsigned char
	normalize(result,result,0,255,NORM_MINMAX,CV_8UC1);

	//显示一张特征脸
	imshow("eigenface1",result);
	waitKey();
}