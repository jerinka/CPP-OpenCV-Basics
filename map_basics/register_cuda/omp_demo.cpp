//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//#include <chrono>
//
//std::chrono::steady_clock::time_point start, stop;
//#define clock_start() {\
//             start = std::chrono::steady_clock::now();\
//             }
//
//#define clock_end() {\
//            stop = std::chrono::steady_clock::now();\
//	    std::chrono::duration<float> time_interval = stop - start;\
//	    std::cout << " Time for color : " << time_interval.count() << " sec \n";\
//            }
//
//int main()
//{
//	Mat src_;
//	const int N = 3;
//	src_ = imread("1_new.png", IMREAD_GRAYSCALE);
//
//	// HostMem allocates page-locked host memory.
//	cuda::HostMem src(src_);
//
//	Mat src1[N];
//	for (int i = 0; i < N; i++)
//		src1[i] = src_;
//
//	cuda::HostMem srca[N];
//	
//
//
//	cuda::GpuMat d_src[N],d_src_;
//	for (int i = 0; i<N; i++)
//	d_src[i] = cuda::GpuMat(src.rows, src.cols, CV_8UC4);
//	d_src_ = cuda::GpuMat(src.rows, src.cols, CV_8UC4);
//	
//
//	cuda::Stream stream[N];
//
//
//
//	for (int threadNum = 0; threadNum < N; threadNum++)
//	{
//		d_src[threadNum].upload(src, stream[threadNum]);
//	}
//
//	clock_start();
//		for (int threadNum = 0; threadNum < N; threadNum++)
//		{
//			cuda::resize(d_src[threadNum], d_src[threadNum], Size(1000, 1000), 0.0, 0.0, 2, stream[threadNum]);
//			cuda::resize(d_src[threadNum], d_src[threadNum], Size(100, 100), 0.0, 0.0, 2, stream[threadNum]);
//		}
//		for (int j = 0; j<N; j++)
//			stream[j].waitForCompletion();  //stream sync
//	clock_end();
//
//
//	clock_start();
//	for (int threadNum = 0; threadNum < N; threadNum++)
//	{
//		cv::resize(src1[threadNum], src1[threadNum],Size(1000, 1000), 0.0, 0.0, 2);
//		cv::resize(src1[threadNum], src1[threadNum], Size(100, 100), 0.0, 0.0, 2);
//	}
//	clock_end();
//
//}