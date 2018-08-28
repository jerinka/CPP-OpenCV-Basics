#include <iostream>
#include "TimeProfiler_.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include <sstream>
using namespace cv;
#include <chrono>
//#include <map>



int main(int argc, char* argv[])
{
	
	std::map< String,int> dic;
	dic["one"] = 1;
	dic["two"] = 3;

	std::map< String, int>::iterator it;

	std::cout << dic["two"] <<std::endl;
	//std::cout << dic["three"] << std::endl;
	it = dic.find("three");
	if(it!=dic.end())
	    std::cout << dic.find("three")->second << std::endl;
	else
		std::cout << "three- not a valid key"<< std::endl;

	std::map<char, int> mymap;

	// first insert function version (single parameter):
	mymap.insert(std::pair<char, int>('a', 100));
	mymap.insert(std::pair<char, int>('z', 200));

	std::pair<std::map<char, int>::iterator, bool> ret;
	ret = mymap.insert(std::pair<char, int>('z', 500));
	if (ret.second == false) {
		std::cout << "element 'z' already existed";
		std::cout << " with a value of " << ret.first->second << '\n';
	}

	std::map< String, TimerElement> timer_dic;
	String name = "imread";
	TimerElement temp(name);
	timer_dic["imread"] = temp;


	TimeProfiler timer;
	timer.start_timer("imread");
	Mat im1 = imread("1_new.png");
	timer.stop_timer("imread");

	timer.start_timer("imread");
	Mat im2 = imread("1_new.png");
	timer.stop_timer("imread");


	timer.get_total_time("imread");
	timer.get_average_time("imread");

	cv::cuda::GpuMat im1_gpu_;
	cv::cuda::GpuMat im2_gpu_;

	cv::cuda::Stream stream[2];

	cv::cuda::GpuMat im1_gpu;
	cv::cuda::GpuMat im2_gpu;

	im1_gpu_.upload(im1, stream[0]);
	im2_gpu_.upload(im2, stream[0]);

	for (int i = 0; i<2; i++)
		stream[i].waitForCompletion();

	//clock_start();
	cuda::cvtColor(im1_gpu_, im1_gpu, CV_BGR2GRAY, 1, stream[0]);
	cuda::cvtColor(im2_gpu_, im2_gpu, CV_BGR2GRAY, 1, stream[1]);

	for (int i = 0; i<2; i++)
		stream[i].waitForCompletion();

	Mat im1g, im2g;
	im1_gpu.download(im1g, stream[0]);
	im2_gpu.download(im2g, stream[1]);

	for (int i = 0; i<2; i++)
		stream[i].waitForCompletion();

	imshow("img1g", im1g);
	imshow("img2g", im2g);

	waitKey(0);

}