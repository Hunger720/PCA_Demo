#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

void main(){

	int nsamples = 0;    //������ͼƬ������
	int w = 92;          //ͼƬ��ȣ��ٶ�����������֪��ͳһ��
	int h = 112;         //ͼƬ�߶ȣ��ٶ�����������֪��ͳһ��

	string datasetfile;  //�������ļ�·��
	string file;         //�����ļ�·��

	ifstream infile;

	Mat img, gray;
	Mat input;           //һ������
	Mat inputs;          //�������ݾ���
	Mat result;

	PCA pca;

	//1.������������
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

		//2.����PCA����ؼ���

		//opencv��PCA������������������Ҫ��Ϊfloat�ͣ���CV_32FC1��CV_64FC1
		inputs.convertTo(inputs,CV_32FC1);

		//���Ĳ���10������ǰ10����������ֵ�����Ӧ����������
		pca(inputs,Mat(),CV_PCA_DATA_AS_ROW,10);

		//��ȡЭ��������������������ת����������ͼƬ����ԭ�ߴ�h*w
		//row(0)��ʾ��ȡ��һ��������ֵ��󣩵���������
		result = pca.eigenvectors.row(0);
		result = result.reshape(result.channels(),h);

		//������Ԫ��ֵ�ķ�Χ�Ӹ���ֵӳ�䵽0~255��������������ת����unsigned char
		normalize(result,result,0,255,NORM_MINMAX,CV_8UC1);

		//��ʾһ��������
		imshow("eigenface1",result);
		waitKey();
	}else{
		cerr<<"error: unable to open input dataset file: "<<infile<<endl;
	}
}