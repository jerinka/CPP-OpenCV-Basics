#include <iostream>
#include "TimeProfiler_.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include <sstream>
using namespace cv;
#include <chrono>
//#include <map>

#include "assert.h"
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cudnn.h"

int main4(int argc, char* argv[])
{
	TimeProfiler timer;

	Mat im1, im1_;
	Mat im2, im2_;

	timer.start("imread");
	im1 = imread("1_new.png");
	im2 = imread("1_new.png");
	timer.stop("imread");

	resize(im1, im1, Size(200, 400));
	resize(im2, im2, Size(200, 400));

	//cudaError_t status = cudaSetDevice(0);
	//std::cout << status << std::endl;

	cv::cuda::GpuMat im1_gpu,im1_gpu_;
	cv::cuda::GpuMat im2_gpu,im2_gpu_;

	cv::cuda::Stream stream[2];

	timer.start("upload");
	im1_gpu.upload(im1);
	im2_gpu.upload(im2);
	timer.stop("upload");

	timer.start("upload_stream");
	im1_gpu.upload(im1, stream[0]);
	im2_gpu.upload(im2, stream[1]);
	for (int i = 0; i<2; i++)
		stream[i].waitForCompletion();
	timer.stop("upload_stream");

	
	for (int k = 0; k < 100; k++)
	{
		timer.start("cvtColor_cpu");
		cv::cvtColor(im1, im1_, CV_BGR2GRAY, 1);
		cv::cvtColor(im2, im2_, CV_BGR2GRAY, 1);
		timer.stop("cvtColor_cpu");
	}

	for (int k = 0; k < 100; k++)
	{
		timer.start("cvtColor_cuda");
		cuda::cvtColor(im1_gpu, im1_gpu_, CV_BGR2GRAY, 1);
		cuda::cvtColor(im2_gpu, im2_gpu_, CV_BGR2GRAY, 1);
		timer.stop("cvtColor_cuda");
	}


	for (int k = 0; k < 100; k++)
	{
		timer.start("cvtColor_cuda_stream");
		cuda::cvtColor(im1_gpu, im1_gpu_, CV_BGR2GRAY, 1, stream[0]);
		cuda::cvtColor(im2_gpu, im2_gpu_, CV_BGR2GRAY, 1, stream[1]);
		for (int i = 0; i < 2; i++)
			stream[i].waitForCompletion();
		timer.stop("cvtColor_cuda_stream");
	}
	

	timer.start("download");
	im1_gpu_.download(im1_, stream[0]);
	im2_gpu_.download(im2_, stream[1]);
	for (int i = 0; i<2; i++)
		stream[i].waitForCompletion();
	timer.stop("download");

	imshow("img1g", im1_);
	imshow("img2g", im2_);

	timer.get_all_average_times();

	waitKey(0);

}