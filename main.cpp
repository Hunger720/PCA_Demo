#include <opencv2/opencv.hpp>
#include <iostream>
#include<stdlib.h>

using namespace cv;
using namespace std;

void main(){

	int nsamples;        //������ͼƬ������
	int w = 92;          //ͼƬ��ȣ��ٶ�����������֪��ͳһ��
	int h = 112;         //ͼƬ�߶ȣ��ٶ�����������֪��ͳһ��
	int index = 0;       //������ţ���0��ʼ

	char filename[150];  //�����ļ�·��

	Mat img, gray;
	Mat inputs;          //�������ݾ���
	Mat result;

	PCA pca;

	//1.������������
	cin>>nsamples;
	inputs.create(nsamples,h*w,CV_8UC1);

	for(int i=0;i<nsamples;i++){

		cin>>filename;
		img = imread(filename);
		cvtColor(img,gray,CV_BGR2GRAY);

		//��һ��h*w����������ת����1*n��������n=h*w����ÿ������Ϊһ��nά����
		//��ת����nά���������������ӵ����������У������ÿһ��Ϊһ������
		for(int j= 0;j<h;j++)
			for(int k=0;k<w;k++){
				inputs.at<uchar>(index,j*w+k) = gray.at<uchar>(j,k);
			}
		index++;
	}

	//2.����PCA����ؼ���

	//opencv��PCA������������������Ҫ��Ϊfloat�ͣ���CV_32FC1��CV_64FC1
	inputs.convertTo(inputs,CV_32FC1);

	//���Ĳ���10������ǰ10����������ֵ�����Ӧ����������
	pca(inputs,Mat(),CV_PCA_DATA_AS_ROW,10);

	//��ȡЭ��������������������ת����������ͼƬ����ԭ�ߴ�h*w
	//row(0)��ʾ��ȡ��һ��������ֵ��󣩵���������
	result = pca.eigenvectors.row(0).reshape(1,h);

	//������Ԫ��ֵ�ķ�Χ�Ӹ���ֵӳ�䵽0~255��������������ת����unsigned char
	normalize(result,result,0,255,NORM_MINMAX,CV_8UC1);

	//��ʾһ��������
	imshow("eigenface1",result);
	waitKey();
}