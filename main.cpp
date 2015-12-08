#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

void main(){

	int nsamples = 0;    //样本（图片）个数
	int w = 92;          //图片宽度（假定所有样本已知且统一）
	int h = 112;         //图片高度（假定所有样本已知且统一）

	string datasetfile;  //样本集文件路径
	string file;         //样本文件路径

	ifstream infile;

	Mat img, gray;
	Mat input;           //一个样本
	Mat inputs;          //样本数据矩阵
	Mat result;

	PCA pca;

	//1.输入样本数据
	cout<<"Input the path of dataset file: ";
	cin>>datasetfile;
	infile.open(datasetfile.c_str());

	if(infile){
		do{
			getline(infile,file);
			img = imread(file);

			cvtColor(img,gray,CV_BGR2GRAY);
			input = gray.reshape(1,1);
			inputs.push_back(input);

			getline(infile,file);
		}while(!infile.eof());

		//2.进行PCA的相关计算

		//opencv的PCA类输入矩阵的数据类型要求为float型，既CV_32FC1或CV_64FC1
		inputs.convertTo(inputs,CV_32FC1);

		//最后的参数10，保留前10个最大的特征值及其对应的特征向量
		pca(inputs,Mat(),CV_PCA_DATA_AS_ROW,10);

		//读取协方差矩阵的特征向量，并转换成样本（图片）的原尺寸h*w
		//row(0)表示读取第一个（特征值最大）的特征向量
		result = pca.eigenvectors.row(0);
		result = result.reshape(result.channels(),h);

		//将向量元素值的范围从浮点值映射到0~255，并将数据类型转换成unsigned char
		normalize(result,result,0,255,NORM_MINMAX,CV_8UC1);

		//显示一张特征脸
		imshow("eigenface1",result);
		waitKey();
	}else{
		cerr<<"error: unable to open input dataset file: "<<infile<<endl;
	}
}