#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <deque>
#include<vector>
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/mat.hpp"
#include <ctime>
#include <chrono>

#include <ctime>

#include <opencv2/core/cuda.hpp>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

#include <opencv2/cudawarping.hpp>

#include <opencv2/cudafeatures2d.hpp>
#include "TimeProfiler_.h"

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cudnn.h"

#include <chrono>
std::chrono::steady_clock::time_point start, stop;
std::string S("");
#define clock_start() {\
             start = std::chrono::steady_clock::now();\
             }
#define clock_end( S ) {\
            stop = std::chrono::steady_clock::now();\
	    std::chrono::duration<float> time_interval = stop - start;\
	    std::cout << S <<": "<< time_interval.count() << " sec \n";\
            }

using namespace std;
using namespace cv;

TimeProfiler timer;

// Specify the number of iterations.
int number_of_iterations = 25;

double termination_eps = 1e-10;

const int imageFlags = INTER_LINEAR + WARP_INVERSE_MAP;
const int maskFlags = INTER_NEAREST + WARP_INVERSE_MAP;


// Define termination criteria
TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations, termination_eps);


class RegisterImagesCpu {

public:
	void image_jacobian_homo_ECC(const Mat& src1, const Mat& src2,
		const Mat& src3, const Mat& src4,
		const Mat& src5, Mat& dst)
	{


		CV_Assert(src1.size() == src2.size());
		CV_Assert(src1.size() == src3.size());
		CV_Assert(src1.size() == src4.size());

		CV_Assert(src1.rows == dst.rows);
		CV_Assert(dst.cols == (src1.cols * 8));
		CV_Assert(dst.type() == CV_32FC1);

		CV_Assert(src5.isContinuous());


		const float* hptr = src5.ptr<float>(0);

		const float h0_ = hptr[0];
		const float h1_ = hptr[3];
		const float h2_ = hptr[6];
		const float h3_ = hptr[1];
		const float h4_ = hptr[4];
		const float h5_ = hptr[7];
		const float h6_ = hptr[2];
		const float h7_ = hptr[5];

		const int w = src1.cols;


		//create denominator for all points as a block
		Mat den_ = src3*h2_ + src4*h5_ + 1.0;//check the time of this! otherwise use addWeighted

											 //create projected points
		Mat hatX_ = -src3*h0_ - src4*h3_ - h6_;
		divide(hatX_, den_, hatX_);
		Mat hatY_ = -src3*h1_ - src4*h4_ - h7_;
		divide(hatY_, den_, hatY_);


		//instead of dividing each block with den,
		//just pre-devide the block of gradients (it's more efficient)

		Mat src1Divided_;
		Mat src2Divided_;

		divide(src1, den_, src1Divided_);
		divide(src2, den_, src2Divided_);


		//compute Jacobian blocks (8 blocks)

		dst.colRange(0, w) = src1Divided_.mul(src3);//1

		dst.colRange(w, 2 * w) = src2Divided_.mul(src3);//2

		Mat temp_ = (hatX_.mul(src1Divided_) + hatY_.mul(src2Divided_));
		dst.colRange(2 * w, 3 * w) = temp_.mul(src3);//3

		hatX_.release();
		hatY_.release();

		dst.colRange(3 * w, 4 * w) = src1Divided_.mul(src4);//4

		dst.colRange(4 * w, 5 * w) = src2Divided_.mul(src4);//5

		dst.colRange(5 * w, 6 * w) = temp_.mul(src4);//6

		src1Divided_.copyTo(dst.colRange(6 * w, 7 * w));//7

		src2Divided_.copyTo(dst.colRange(7 * w, 8 * w));//8
	}


	void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst)
	{
		/* this functions is used for two types of projections. If src1.cols ==src.cols
		it does a blockwise multiplication (like in the outer product of vectors)
		of the blocks in matrices src1 and src2 and dst
		has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
		(number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

		The number_of_blocks is equal to the number of parameters we are lloking for
		(i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)

		*/
		CV_Assert(src1.rows == src2.rows);
		CV_Assert((src1.cols % src2.cols) == 0);
		int w;

		float* dstPtr = dst.ptr<float>(0);

		if (src1.cols != src2.cols) {//dst.cols==1
			w = src2.cols;
			for (int i = 0; i < dst.rows; i++) {
				dstPtr[i] = (float)src2.dot(src1.colRange(i*w, (i + 1)*w));
			}
		}

		else {
			CV_Assert(dst.cols == dst.rows); //dst is square (and symmetric)
			w = src2.cols / dst.cols;
			Mat mat;
			for (int i = 0; i < dst.rows; i++) {

				mat = Mat(src1.colRange(i*w, (i + 1)*w));
				dstPtr[i*(dst.rows + 1)] = (float)pow(norm(mat), 2); //diagonal elements

				for (int j = i + 1; j < dst.cols; j++) { //j starts from i+1
					dstPtr[i*dst.cols + j] = (float)mat.dot(src2.colRange(j*w, (j + 1)*w));
					dstPtr[j*dst.cols + i] = dstPtr[i*dst.cols + j]; //due to symmetry
				}
			}
		}
	}

	void update_warping_matrix_ECC(Mat& map_matrix, const Mat& update, const int motionType)
	{
		CV_Assert(map_matrix.type() == CV_32FC1);
		CV_Assert(update.type() == CV_32FC1);

		CV_Assert(motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN ||
			motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY);

		if (motionType == MOTION_HOMOGRAPHY)
			CV_Assert(map_matrix.rows == 3 && update.rows == 8);
		else if (motionType == MOTION_AFFINE)
			CV_Assert(map_matrix.rows == 2 && update.rows == 6);
		else if (motionType == MOTION_EUCLIDEAN)
			CV_Assert(map_matrix.rows == 2 && update.rows == 3);
		else
			CV_Assert(map_matrix.rows == 2 && update.rows == 2);

		CV_Assert(update.cols == 1);

		CV_Assert(map_matrix.isContinuous());
		CV_Assert(update.isContinuous());


		float* mapPtr = map_matrix.ptr<float>(0);
		const float* updatePtr = update.ptr<float>(0);


		if (motionType == MOTION_TRANSLATION) {
			mapPtr[2] += updatePtr[0];
			mapPtr[5] += updatePtr[1];
		}
		if (motionType == MOTION_AFFINE) {
			mapPtr[0] += updatePtr[0];
			mapPtr[3] += updatePtr[1];
			mapPtr[1] += updatePtr[2];
			mapPtr[4] += updatePtr[3];
			mapPtr[2] += updatePtr[4];
			mapPtr[5] += updatePtr[5];
		}
		if (motionType == MOTION_HOMOGRAPHY) {
			mapPtr[0] += updatePtr[0];
			mapPtr[3] += updatePtr[1];
			mapPtr[6] += updatePtr[2];
			mapPtr[1] += updatePtr[3];
			mapPtr[4] += updatePtr[4];
			mapPtr[7] += updatePtr[5];
			mapPtr[2] += updatePtr[6];
			mapPtr[5] += updatePtr[7];
		}
		if (motionType == MOTION_EUCLIDEAN) {
			double new_theta = updatePtr[0];
			new_theta += asin(mapPtr[3]);

			mapPtr[2] += updatePtr[1];
			mapPtr[5] += updatePtr[2];
			mapPtr[0] = mapPtr[4] = (float)cos(new_theta);
			mapPtr[3] = (float)sin(new_theta);
			mapPtr[1] = -mapPtr[3];
		}
	}

	int registerimg(Mat im1, Mat im2, Mat inputMask, Mat im2_aligned)
	{

		Mat im1_gray, im2_gray;
		cvtColor(im1, im1_gray, CV_BGR2GRAY);

		cvtColor(im2, im2_gray, CV_BGR2GRAY);

		const int motionType = MOTION_HOMOGRAPHY;

		// Set a 2x3 or 3x3 warp matrix depending on the motion model.

		Mat warpMatrix;
		// Initialize the matrix to identity
		if (motionType == MOTION_HOMOGRAPHY)
			warpMatrix = Mat::eye(3, 3, CV_32F);
		else
			warpMatrix = Mat::eye(2, 3, CV_32F);


		Mat src_persp;

		Mat src = im1_gray.clone();

		Mat dst = im2_gray.clone();

		Mat map = warpMatrix.clone();

		CV_Assert(!src.empty());
		CV_Assert(!dst.empty());

		if (map.empty())
		{
			int rowCount = 2;
			if (motionType == MOTION_HOMOGRAPHY)
				rowCount = 3;

			warpMatrix.create(rowCount, 3, CV_32FC1);
			map = warpMatrix.clone();
			map = Mat::eye(rowCount, 3, CV_32F);
		}

		if (!(src.type() == dst.type()))
			CV_Error(Error::StsUnmatchedFormats, "Both input images must have the same data type");

		//accept only 1-channel images
		if (src.type() != CV_8UC1 && src.type() != CV_32FC1)
			CV_Error(Error::StsUnsupportedFormat, "Images must have 8uC1 or 32fC1 type");

		if (map.type() != CV_32FC1)
			CV_Error(Error::StsUnsupportedFormat, "warpMatrix must be single-channel floating-point matrix");

		CV_Assert(map.cols == 3);
		CV_Assert(map.rows == 2 || map.rows == 3);

		CV_Assert(motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY || motionType == MOTION_EUCLIDEAN || motionType == MOTION_TRANSLATION);
		if (motionType == MOTION_HOMOGRAPHY)
		{
			CV_Assert(map.rows == 3);
		}

		CV_Assert(criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
		const int    numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
		const double termination_eps = (criteria.type & TermCriteria::EPS) ? criteria.epsilon : -1;
		int paramTemp = 6;//default: affine
		switch (motionType)
		{
		case MOTION_TRANSLATION:
			paramTemp = 2;
			break;
		case MOTION_EUCLIDEAN:
			paramTemp = 3;
			break;
		case MOTION_HOMOGRAPHY:
			paramTemp = 8;
			break;
		}
		const int numberOfParameters = paramTemp;

		const int ws = src.cols;
		const int hs = src.rows;
		const int wd = dst.cols;
		const int hd = dst.rows;

		Mat Xcoord = Mat(1, ws, CV_32F);
		Mat Ycoord = Mat(hs, 1, CV_32F);
		Mat Xgrid = Mat(hs, ws, CV_32F);
		Mat Ygrid = Mat(hs, ws, CV_32F);

		float* XcoPtr = Xcoord.ptr<float>(0);
		float* YcoPtr = Ycoord.ptr<float>(0);
		int j;
		for (j = 0; j < ws; j++)
			XcoPtr[j] = (float)j;
		for (j = 0; j < hs; j++)
			YcoPtr[j] = (float)j;

		repeat(Xcoord, hs, 1, Xgrid);
		repeat(Ycoord, 1, ws, Ygrid);

		Xcoord.release();
		Ycoord.release();
		Mat templateZM = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
		Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
		Mat imageFloat = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
		Mat imageWarped = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image
		Mat imageMask = Mat(hs, ws, CV_8U); //to store the final mask

		Mat preMask;
		int start_gb;
		int stop_gb;
		if (inputMask.empty())
			preMask = Mat::ones(hd, wd, CV_8U);
		else

			threshold(inputMask, preMask, 0, 1, THRESH_BINARY);
		src.convertTo(templateFloat, templateFloat.type());

		GaussianBlur(templateFloat, templateFloat, Size(5, 5), 0, 0);


		Mat preMaskFloat;
		preMask.convertTo(preMaskFloat, CV_32F);

		GaussianBlur(preMaskFloat, preMaskFloat, Size(5, 5), 0, 0);

		// Change threshold.
		preMaskFloat *= (0.5 / 0.95);
		// Rounding conversion.
		preMaskFloat.convertTo(preMask, preMask.type());

		preMask.convertTo(preMaskFloat, preMaskFloat.type());

		dst.convertTo(imageFloat, imageFloat.type());

		GaussianBlur(imageFloat, imageFloat, Size(5, 5), 0, 0);


		// needed matrices for gradients and warped gradients
		Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
		Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);
		Mat gradientXWarped = Mat(hs, ws, CV_32FC1);
		Mat gradientYWarped = Mat(hs, ws, CV_32FC1);


		// calculate first order image derivatives
		Matx13f dx(-0.5f, 0.0f, 0.5f);

		filter2D(imageFloat, gradientX, -1, dx);
		filter2D(imageFloat, gradientY, -1, dx.t());
		gradientX = gradientX.mul(preMaskFloat);
		gradientY = gradientY.mul(preMaskFloat);


		// matrices needed for solving linear equation system for maximizing ECC
		Mat jacobian = Mat(hs, ws*numberOfParameters, CV_32F);
		Mat hessian = Mat(numberOfParameters, numberOfParameters, CV_32F);
		Mat hessianInv = Mat(numberOfParameters, numberOfParameters, CV_32F);
		Mat imageProjection = Mat(numberOfParameters, 1, CV_32F);
		Mat templateProjection = Mat(numberOfParameters, 1, CV_32F);
		Mat imageProjectionHessian = Mat(numberOfParameters, 1, CV_32F);
		Mat errorProjection = Mat(numberOfParameters, 1, CV_32F);

		Mat deltaP = Mat(numberOfParameters, 1, CV_32F);//transformation parameter correction
		Mat error = Mat(hs, ws, CV_32F);//error as 2D matrix

		// iteratively update map_matrix
		double rho = -1;
		double last_rho = -termination_eps;


		for (int i = 1; (i <= numberOfIterations) && (fabs(rho - last_rho) >= termination_eps); i++)
		{

			if (motionType == MOTION_HOMOGRAPHY)

			{
				warpPerspective(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
				warpPerspective(gradientX, gradientXWarped, map, gradientXWarped.size(), imageFlags);
				warpPerspective(gradientY, gradientYWarped, map, gradientYWarped.size(), imageFlags);
				warpPerspective(preMask, imageMask, map, imageMask.size(), maskFlags);


			}

			Scalar imgMean, imgStd, tmpMean, tmpStd;
			meanStdDev(imageWarped, imgMean, imgStd, imageMask);
			meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);
			subtract(imageWarped, imgMean, imageWarped, imageMask);//zero-mean input	
			templateZM = Mat::zeros(templateZM.rows, templateZM.cols, templateZM.type());
			subtract(templateFloat, tmpMean, templateZM, imageMask);//zero-mean template


			const double tmpNorm = std::sqrt(countNonZero(imageMask)*(tmpStd.val[0])*(tmpStd.val[0]));
			const double imgNorm = std::sqrt(countNonZero(imageMask)*(imgStd.val[0])*(imgStd.val[0]));

			// calculate jacobian of image wrt parameters

			if (motionType == MOTION_HOMOGRAPHY)

				image_jacobian_homo_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);

			// calculate Hessian and its inverse

			project_onto_jacobian_ECC(jacobian, jacobian, hessian);

			hessianInv = hessian.inv();

			const double correlation = templateZM.dot(imageWarped);

			// calculate enhanced correlation coefficiont (ECC)->rho
			last_rho = rho;

			rho = correlation / (imgNorm*tmpNorm);

			if (cvIsNaN(rho)) {
				CV_Error(Error::StsNoConv, "NaN encountered.");
			}

			// project images into jacobian

			project_onto_jacobian_ECC(jacobian, imageWarped, imageProjection);

			project_onto_jacobian_ECC(jacobian, templateZM, templateProjection);


			// calculate the parameter lambda to account for illumination variation
			imageProjectionHessian = hessianInv*imageProjection;
			const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
			const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
			if (lambda_d <= 0.0)
			{
				rho = -1;
				CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");

			}
			const double lambda = (lambda_n / lambda_d);

			// estimate the update step delta_p
			error = lambda*templateZM - imageWarped;

			project_onto_jacobian_ECC(jacobian, error, errorProjection);

			deltaP = hessianInv * errorProjection;


			// update warping matrix
			update_warping_matrix_ECC(map, deltaP, motionType);

		}


		if (motionType != MOTION_HOMOGRAPHY)
			// Use warpAffine for Translation, Euclidean and Affine
			warpAffine(im2, im2_aligned, map, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);
		else
			// Use warpPerspective for Homography

			warpPerspective(im2, im2_aligned, map, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);


		// Show final result
		imshow("Image 1", im1);
		imshow("Image 2", im2);
		imshow("Image 2 Aligned", im2_aligned);

		return(rho);
	}
};



class RegisterImagesCuda {
public:
	RegisterImagesCuda(int h, int w) : hs(h), ws(w), hd(h), wd(w)
	{
		cuda::setDevice(0);


		templateZM_gpu = cv::cuda::GpuMat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
		templateFloat_gpu = cv::cuda::GpuMat(hs, ws, CV_32F);// to store the (smoothed) template
		imageFloat_gpu = cv::cuda::GpuMat(hs, ws, CV_32F);// to store the (smoothed) input image
		imageWarped_gpu = cv::cuda::GpuMat(hs, ws, CV_32F);// to store the warped zero-mean input image
		imageMask_gpu = cv::cuda::GpuMat(hs, ws, CV_8U); //to store the final mask

		if (motionType == MOTION_HOMOGRAPHY)
			warpMatrix = Mat::eye(3, 3, CV_32F);
		else
			warpMatrix = Mat::eye(2, 3, CV_32F);

		map = warpMatrix.clone();


		// If the user passed an un-initialized warpMatrix, initialize to identity
		if (map.empty()) {
			int rowCount = 2;
			if (motionType == MOTION_HOMOGRAPHY)
				rowCount = 3;

			warpMatrix.create(rowCount, 3, CV_32FC1);
			map = warpMatrix.clone();
			map = Mat::eye(rowCount, 3, CV_32F);
		}

		warpMatrix_gpu.upload(warpMatrix);


		map_gpu.upload(map);

		if (!(src_gpu.type() == dst_gpu.type()))
			CV_Error(Error::StsUnmatchedFormats, "Both input images must have the same data type");

		//accept only 1-channel images
		if (src_gpu.type() != CV_8UC1 && src_gpu.type() != CV_32FC1)
			CV_Error(Error::StsUnsupportedFormat, "Images must have 8uC1 or 32fC1 type");

		if (map_gpu.type() != CV_32FC1)
			CV_Error(Error::StsUnsupportedFormat, "warpMatrix must be single-channel floating-point matrix");

		CV_Assert(map_gpu.cols == 3);
		CV_Assert(map_gpu.rows == 2 || map_gpu.rows == 3);

		CV_Assert(motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY ||
			motionType == MOTION_EUCLIDEAN || motionType == MOTION_TRANSLATION);

		if (motionType == MOTION_HOMOGRAPHY) {
			CV_Assert(map_gpu.rows == 3);
		}


		CV_Assert(criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);

		//switch (motionType) {
		//case MOTION_TRANSLATION:
		//	paramTemp = 2;
		//	break;
		//case MOTION_EUCLIDEAN:
		//	paramTemp = 3;
		//	break;
		//case MOTION_HOMOGRAPHY:
		//	paramTemp = 8;
		//	break;
		//}

		//const int numberOfParameters = paramTemp;



		Xcoord = Mat(1, ws, CV_32F);
		Ycoord = Mat(hs, 1, CV_32F);
		Xgrid = Mat(hs, ws, CV_32F);
		Ygrid = Mat(hs, ws, CV_32F);


		float* XcoPtr = Xcoord.ptr<float>(0);
		float* YcoPtr = Ycoord.ptr<float>(0);
		int j;

		for (j = 0; j < ws; j++)
			XcoPtr[j] = (float)j;
		for (j = 0; j < hs; j++)
			YcoPtr[j] = (float)j;

		repeat(Xcoord, hs, 1, Xgrid);
		repeat(Ycoord, 1, ws, Ygrid);

		Xcoord.release();
		Ycoord.release();

	}
	const int hs, ws;
	const int hd, wd;
	cv::cuda::Stream stream[5];

	const int motionType = MOTION_HOMOGRAPHY;

	cv::cuda::GpuMat im1_gpu;
	cv::cuda::GpuMat im2_gpu;
	cv::cuda::GpuMat inputMask_gpu;

	cv::cuda::GpuMat im1_gray_gpu;
	cv::cuda::GpuMat im2_gray_gpu;

	cv::cuda::GpuMat src_gpu;
	cv::cuda::GpuMat dst_gpu;

	cv::cuda::GpuMat warpMatrix_gpu;

	cv::cuda::GpuMat templateZM_gpu;// to store the (smoothed)zero-mean version of template
	cv::cuda::GpuMat templateFloat_gpu;// to store the (smoothed) template
	cv::cuda::GpuMat imageFloat_gpu;// to store the (smoothed) input image
	cv::cuda::GpuMat imageWarped_gpu;// to store the warped zero-mean input image
	cv::cuda::GpuMat imageMask_gpu; //to store the final mask

	cv::Mat templateImage, inputImage;
	Mat  src, dst;
	Mat templateZM, templateFloat, imageFloat, imageWarped, imageMask, preMask, preMaskFloat;
	Mat gradientXWarped, gradientYWarped;
	Mat jacobian, deltaP, error;

	cv::cuda::GpuMat map_gpu;
	Mat map;
	Mat warpMatrix;

	const int    numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
	const double termination_eps = (criteria.type & TermCriteria::EPS) ? criteria.epsilon : -1;

	int paramTemp = 6;//default: affine


	cv::cuda::GpuMat Xgrid_gpu;
	cv::cuda::GpuMat Ygrid_gpu;

	const int numberOfParameters = paramTemp;

	Mat Xcoord ;
	Mat Ycoord;
	Mat Xgrid ;
	Mat Ygrid ;


	void image_jacobian_homo_ECC(const cv::cuda::GpuMat& src1_gpu, const cv::cuda::GpuMat& src2_gpu,
		const cv::cuda::GpuMat& src3_gpu, const cv::cuda::GpuMat& src4_gpu,
		const Mat& src5, Mat& dst)
	{

		Mat src1Divided_, src2Divided_, temp_;

		CV_Assert(src1_gpu.size() == src2_gpu.size());
		CV_Assert(src1_gpu.size() == src3_gpu.size());
		CV_Assert(src1_gpu.size() == src4_gpu.size());

		CV_Assert(src1_gpu.rows == dst.rows);

		CV_Assert(dst.cols == (src1_gpu.cols * 8));
		CV_Assert(dst.type() == CV_32FC1);

		CV_Assert(src5.isContinuous());


		const float* hptr = src5.ptr<float>(0);

		const float h0_ = hptr[0];
		const float h1_ = hptr[3];
		const float h2_ = hptr[6];
		const float h3_ = hptr[1];
		const float h4_ = hptr[4];
		const float h5_ = hptr[7];
		const float h6_ = hptr[2];
		const float h7_ = hptr[5];

		const int w = src1_gpu.cols;


		//create denominator for all points as a block
		//Mat den_ = src3*h2_ + src4*h5_ + 1.0;//check the time of this! otherwise use addWeighted
		Mat den_;

		cv::cuda::GpuMat den1_gpu;
		cv::cuda::GpuMat den2_gpu;
		cv::cuda::GpuMat den3_gpu;
		cv::cuda::GpuMat den__gpu;

		cv::cuda::multiply(src3_gpu, h2_, den1_gpu);
		cv::cuda::multiply(src4_gpu, h5_, den2_gpu);
		cv::cuda::add(den1_gpu, den2_gpu, den3_gpu);
		cv::cuda::add(den3_gpu, 1, den__gpu);



		//create projected points
		//Mat hatX_ = -src3*h0_ - src4*h3_ - h6_;

		cv::cuda::GpuMat hatX1_gpu;
		cv::cuda::GpuMat hatX2_gpu;
		cv::cuda::GpuMat hatX3_gpu;
		cv::cuda::GpuMat hatX4_gpu;
		cv::cuda::GpuMat hatX__gpu;

		cuda::multiply(src3_gpu, -1, hatX1_gpu);
		cuda::multiply(hatX1_gpu, h0_, hatX2_gpu);
		cuda::multiply(src4_gpu, h3_, hatX3_gpu);


		cuda::subtract(hatX2_gpu, hatX3_gpu, hatX4_gpu);
		cuda::subtract(hatX4_gpu, h6_, hatX__gpu);

		cv::cuda::divide(hatX__gpu, den__gpu, hatX__gpu);

		cv::cuda::GpuMat hatY__gpu;
		cv::cuda::GpuMat hatY1_gpu;
		cv::cuda::GpuMat hatY2_gpu;
		cv::cuda::GpuMat hatY3_gpu;
		cv::cuda::GpuMat hatY4_gpu;


		cuda::multiply(src3_gpu, -1, hatY1_gpu);
		cuda::multiply(hatY1_gpu, h1_, hatY2_gpu);
		cuda::multiply(src4_gpu, h4_, hatY3_gpu);


		cuda::subtract(hatY2_gpu, hatY3_gpu, hatY4_gpu);
		cuda::subtract(hatY4_gpu, h7_, hatY__gpu);

		cv::cuda::divide(hatY__gpu, den__gpu, hatY__gpu);

		//instead of dividing each block with den,
		//just pre-devide the block of gradients (it's more efficient)

		cv::cuda::GpuMat src1Divided__gpu;
		cv::cuda::GpuMat src2Divided__gpu;

		cv::cuda::GpuMat dst1_gpu;
		cv::cuda::GpuMat dst2_gpu;
		cv::cuda::GpuMat dst3_gpu;
		cv::cuda::GpuMat dst4_gpu;
		cv::cuda::GpuMat dst5_gpu;
		cv::cuda::GpuMat dst6_gpu;
		cv::cuda::GpuMat dst_gpu;
		cv::cuda::GpuMat temp__gpu;
		cv::cuda::GpuMat temp1_gpu;
		cv::cuda::GpuMat temp2_gpu;


		dst_gpu.upload(dst);

		cv::cuda::divide(src1_gpu, den__gpu, src1Divided__gpu);
		cv::cuda::divide(src2_gpu, den__gpu, src2Divided__gpu);

		cv::cuda::GpuMat roi_gpu1(dst_gpu(cv::Rect(0, 0, w, dst_gpu.rows)));
		cv::cuda::multiply(src1Divided__gpu, src3_gpu, dst1_gpu);
		dst1_gpu.copyTo(roi_gpu1);

		//dst.colRange(w,2*w) = src2Divided_.mul(src3);//2
		cv::cuda::GpuMat roi_gpu2(dst_gpu(cv::Rect(w, 0, w, dst_gpu.rows)));
		cv::cuda::multiply(src2Divided__gpu, src3_gpu, dst2_gpu);
		dst2_gpu.copyTo(roi_gpu2);

		//Mat temp_ = (hatX_.mul(src1Divided_)+hatY_.mul(src2Divided_));

		cv::cuda::multiply(hatX__gpu, src1Divided__gpu, temp1_gpu);
		cv::cuda::multiply(hatY__gpu, src2Divided__gpu, temp2_gpu);
		cv::cuda::add(temp1_gpu, temp2_gpu, temp__gpu);


		//dst.colRange(2*w,3*w) = temp_.mul(src3);//3
		cv::cuda::GpuMat roi_gpu3(dst_gpu(cv::Rect(2 * w, 0, w, dst_gpu.rows)));
		cv::cuda::multiply(temp__gpu, src3_gpu, dst3_gpu);
		dst3_gpu.copyTo(roi_gpu3);

		//dst.colRange(3*w, 4*w) = src1Divided_.mul(src4);//4
		cv::cuda::GpuMat roi_gpu4(dst_gpu(cv::Rect(3 * w, 0, w, dst_gpu.rows)));
		cv::cuda::multiply(src1Divided__gpu, src4_gpu, dst4_gpu);
		dst4_gpu.copyTo(roi_gpu4);

		//dst.colRange(4*w, 5*w) = src2Divided_.mul(src4);//5
		cv::cuda::GpuMat roi_gpu5(dst_gpu(cv::Rect(4 * w, 0, w, dst_gpu.rows)));
		cv::cuda::multiply(src2Divided__gpu, src4_gpu, dst5_gpu);
		dst5_gpu.copyTo(roi_gpu5);

		//dst.colRange(5*w, 6*w) = temp_.mul(src4);//6
		cv::cuda::GpuMat roi_gpu6(dst_gpu(cv::Rect(5 * w, 0, w, dst_gpu.rows)));
		cv::cuda::multiply(temp__gpu, src4_gpu, dst6_gpu);
		dst6_gpu.copyTo(roi_gpu6);


		src1Divided__gpu.copyTo(dst_gpu.colRange(6 * w, 7 * w));//7

		src2Divided__gpu.copyTo(dst_gpu.colRange(7 * w, 8 * w));//8

		dst_gpu.download(dst);

	}

	void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst)
	{
		/* this functions is used for two types of projections. If src1.cols ==src.cols
		it does a blockwise multiplication (like in the outer product of vectors)
		of the blocks in matrices src1 and src2 and dst
		has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
		(number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

		The number_of_blocks is equal to the number of parameters we are lloking for
		(i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)

		*/
		CV_Assert(src1.rows == src2.rows);
		CV_Assert((src1.cols % src2.cols) == 0);


		int w;
		Mat dst_new2;
		float* dstPtr = dst.ptr<float>(0);


		cv::cuda::GpuMat src1_gpu;
		cv::cuda::GpuMat src2_gpu;
		cv::cuda::GpuMat dst1_gpu;
		cv::cuda::GpuMat dst2_gpu;
		cv::cuda::GpuMat dst_gpu;
		Mat dst1, dst2;
		src1_gpu.upload(src1);
		src2_gpu.upload(src2);
		dst_gpu.upload(dst);


		if (src1_gpu.cols != src2_gpu.cols) {//dst.cols==1
											 //cout<<"test2"<<endl;
			w = src2_gpu.cols;
			for (int i = 0; i < dst_gpu.rows; i++) {

				//dstPtr[i] = (float) src2.dot(src1.colRange(i*w,(i+1)*w));

				///with mult and sum/////
				cv::cuda::GpuMat roi_gpu11(src1_gpu, cv::Rect(i*w, 0, w, src1_gpu.rows));

				cv::cuda::multiply(src2_gpu, roi_gpu11, dst1_gpu);
				Scalar sum = cuda::sum(dst1_gpu);
				dstPtr[i] = sum[0];
				//cout<<"dstptr_if_sum_gpu "<< dstPtr[i]<<endl;

			}

		}


		else {

			CV_Assert(dst_gpu.cols == dst_gpu.rows); //dst is square (and symmetric)
			w = src2_gpu.cols / dst_gpu.cols;

			for (int i = 0; i < dst_gpu.rows; i++) {

				//mat = Mat(src1.colRange(i*w, (i+1)*w));
				cv::cuda::GpuMat mat_gpu(src1_gpu(cv::Rect(i*w, 0, w, src1_gpu.rows)));
				dstPtr[i*(dst_gpu.rows + 1)] = (float)pow(cuda::norm(mat_gpu, NORM_L2), 2); //diagonal elements
																							//cout<<"power_gpu"<<pow(cuda::norm(mat_gpu,NORM_L2),2)<<"\n";
				for (int j = i + 1; j < dst_gpu.cols; j++) { //j starts from i+1
															 //dstPtr[i*dst.cols+j] = (float) mat.dot(src2.colRange(j*w, (j+1)*w));


															 ///with mult and sum/////
					cv::cuda::GpuMat roi_gpu12(src2_gpu, cv::Rect(j*w, 0, w, src2_gpu.rows));

					cv::cuda::multiply(mat_gpu, roi_gpu12, dst2_gpu);

					Scalar sum2 = cuda::sum(dst2_gpu);
					dstPtr[i*dst_gpu.cols + j] = sum2[0];


					dstPtr[j*dst_gpu.cols + i] = dstPtr[i*dst_gpu.cols + j]; //due to symmetry
																			 //cout<<"dstptr_else_gpu"<<dstPtr[10]<<endl;
				}
			}

		}



	}

	void update_warping_matrix_ECC(Mat& map_matrix, const Mat& update, const int motionType)
	{
		CV_Assert(map_matrix.type() == CV_32FC1);
		CV_Assert(update.type() == CV_32FC1);

		CV_Assert(motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN ||
			motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY);

		if (motionType == MOTION_HOMOGRAPHY)
			CV_Assert(map_matrix.rows == 3 && update.rows == 8);
		else if (motionType == MOTION_AFFINE)
			CV_Assert(map_matrix.rows == 2 && update.rows == 6);
		else if (motionType == MOTION_EUCLIDEAN)
			CV_Assert(map_matrix.rows == 2 && update.rows == 3);
		else
			CV_Assert(map_matrix.rows == 2 && update.rows == 2);

		CV_Assert(update.cols == 1);

		CV_Assert(map_matrix.isContinuous());
		CV_Assert(update.isContinuous());


		float* mapPtr = map_matrix.ptr<float>(0);
		const float* updatePtr = update.ptr<float>(0);


		if (motionType == MOTION_TRANSLATION) {
			mapPtr[2] += updatePtr[0];
			mapPtr[5] += updatePtr[1];
		}
		if (motionType == MOTION_AFFINE) {
			mapPtr[0] += updatePtr[0];
			mapPtr[3] += updatePtr[1];
			mapPtr[1] += updatePtr[2];
			mapPtr[4] += updatePtr[3];
			mapPtr[2] += updatePtr[4];
			mapPtr[5] += updatePtr[5];
		}
		if (motionType == MOTION_HOMOGRAPHY) {
			mapPtr[0] += updatePtr[0];
			mapPtr[3] += updatePtr[1];
			mapPtr[6] += updatePtr[2];
			mapPtr[1] += updatePtr[3];
			mapPtr[4] += updatePtr[4];
			mapPtr[7] += updatePtr[5];
			mapPtr[2] += updatePtr[6];
			mapPtr[5] += updatePtr[7];
		}
		if (motionType == MOTION_EUCLIDEAN) {
			double new_theta = updatePtr[0];
			new_theta += asin(mapPtr[3]);

			mapPtr[2] += updatePtr[1];
			mapPtr[5] += updatePtr[2];
			mapPtr[0] = mapPtr[4] = (float)cos(new_theta);
			mapPtr[3] = (float)sin(new_theta);
			mapPtr[1] = -mapPtr[3];
		}
	}

	int registerimg(Mat & im1, Mat & im2, Mat & inputMask, Mat & im2_aligned)
	{
		timer.start("upload");
		im1_gpu.upload(im1, stream[0]);
		timer.stop("upload");
		timer.start("upload");
		im2_gpu.upload(im2, stream[1]);
		timer.stop("upload");
		timer.start("upload");
		inputMask_gpu.upload(inputMask, stream[2]);
		timer.stop("upload");

		timer.start("cvtColor_gpu");
		cuda::cvtColor(im1_gpu, im1_gray_gpu, COLOR_RGB2GRAY, 1, stream[0]);
		timer.stop("cvtColor_gpu");
		timer.start("cvtColor_gpu");
		cuda::cvtColor(im2_gpu, im2_gray_gpu, COLOR_RGB2GRAY, 1, stream[1]);
		timer.stop("cvtColor_gpu");



		src_gpu = im1_gray_gpu;
		dst_gpu = im2_gray_gpu;

		CV_Assert(!src_gpu.empty());
		CV_Assert(!dst_gpu.empty());


		cv::cuda::GpuMat preMask_gpu;
		if (inputMask_gpu.empty())
			preMask_gpu = cv::cuda::GpuMat(hd, wd, CV_8UC1, 1);
		else
		{
			timer.start("threshold_gpu");
			cv::cuda::threshold(inputMask_gpu, preMask_gpu, 0, 1, THRESH_BINARY);
			timer.stop("threshold_gpu");
		}
		//gaussian filtering is optional
		src_gpu.convertTo(templateFloat_gpu, templateFloat_gpu.type());
		Ptr<cuda::Filter> gaussianblur_filter1 = cv::cuda::createGaussianFilter(templateFloat_gpu.type(), templateFloat_gpu.type(), Size(5, 5), 0, 0);


		cv::cuda::GpuMat preMaskFloat_gpu;
		preMask_gpu.convertTo(preMaskFloat_gpu, CV_32F);
		//GaussianBlur(preMaskFloat, preMaskFloat, Size(5, 5), 0, 0);
		Ptr<cuda::Filter> gaussianblur_filter2 = cv::cuda::createGaussianFilter(preMaskFloat_gpu.type(), preMaskFloat_gpu.type(), Size(5, 5), 0, 0);


		// Change threshold.
		preMaskFloat_gpu.convertTo(preMaskFloat_gpu, preMaskFloat_gpu.type(), (0.5 / 0.95));
		// Rounding conversion.
		preMaskFloat_gpu.convertTo(preMask_gpu, preMask_gpu.type());
		preMask_gpu.convertTo(preMaskFloat_gpu, preMaskFloat_gpu.type());

		dst_gpu.convertTo(imageFloat_gpu, imageFloat_gpu.type());
		//GaussianBlur(imageFloat, imageFloat, Size(5, 5), 0, 0);
		Ptr<cuda::Filter> gaussianblur_filter3 = cv::cuda::createGaussianFilter(imageFloat_gpu.type(), imageFloat_gpu.type(), Size(5, 5), 0, 0);

		timer.start("gaussian_gpu");
		gaussianblur_filter1->apply(templateFloat_gpu, templateFloat_gpu, stream[0]);
		timer.stop("gaussian_gpu");
		timer.start("gaussian_gpu");
		gaussianblur_filter2->apply(preMaskFloat_gpu, preMaskFloat_gpu, stream[1]);
		timer.stop("gaussian_gpu");
		timer.start("gaussian_gpu");
		gaussianblur_filter3->apply(imageFloat_gpu, imageFloat_gpu, stream[2]);
		timer.stop("gaussian_gpu");

		for (int i_ = 0; i_ < 3; i_++)
			stream[i_].waitForCompletion();  //stream sync



											 // needed matrices for gradients and warped gradients
		Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
		Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);
		cv::cuda::GpuMat gradientXWarped_gpu = cv::cuda::GpuMat(hs, ws, CV_32FC1);
		cv::cuda::GpuMat gradientYWarped_gpu = cv::cuda::GpuMat(hs, ws, CV_32FC1);

		imageFloat_gpu.download(imageFloat);


		// calculate first order image derivatives
		Matx13f dx(-0.5f, 0.0f, 0.5f);

		filter2D(imageFloat, gradientX, -1, dx);
		filter2D(imageFloat, gradientY, -1, dx.t());

		imageFloat_gpu.upload(imageFloat);

		cv::cuda::GpuMat gradientX_gpu;
		gradientX_gpu.upload(gradientX);
		cv::cuda::GpuMat gradientY_gpu;
		gradientY_gpu.upload(gradientY);


		//gradientX = gradientX.mul(preMaskFloat);
		timer.start("multiply_gpu");
		cv::cuda::multiply(gradientX_gpu, preMaskFloat_gpu, gradientX_gpu, 1, -1, stream[0]);
		timer.stop("multiply_gpu");

		//gradientY = gradientY.mul(preMaskFloat);
		timer.start("multiply_gpu");
		cv::cuda::multiply(gradientY_gpu, preMaskFloat_gpu, gradientY_gpu, 1, -1, stream[1]);
		timer.stop("multiply_gpu");

		// matrices needed for solving linear equation system for maximizing ECC
		cv::cuda::GpuMat jacobian_gpu = cv::cuda::GpuMat(hs, ws*numberOfParameters, CV_32F);

		Mat hessian = Mat(numberOfParameters, numberOfParameters, CV_32F);
		Mat hessianInv = Mat(numberOfParameters, numberOfParameters, CV_32F);

		Mat imageProjection = Mat(numberOfParameters, 1, CV_32F);

		Mat templateProjection = Mat(numberOfParameters, 1, CV_32F);

		Mat imageProjectionHessian = Mat(numberOfParameters, 1, CV_32F);

		Mat errorProjection = Mat(numberOfParameters, 1, CV_32F);

		cv::cuda::GpuMat deltaP_gpu = cv::cuda::GpuMat(numberOfParameters, 1, CV_32F);//transformation parameter correction
		cv::cuda::GpuMat error_gpu = cv::cuda::GpuMat(hs, ws, CV_32F);//error as 2D matrix

		const int imageFlags = INTER_LINEAR + WARP_INVERSE_MAP;
		const int maskFlags = INTER_NEAREST + WARP_INVERSE_MAP;



		map_gpu.download(map);


		// iteratively update map_matrix
		double rho = -1;
		double last_rho = -termination_eps;
		for (int i = 1; (i <= numberOfIterations) && (fabs(rho - last_rho) >= termination_eps); i++)
		{


			if (motionType == MOTION_HOMOGRAPHY)
			{
				timer.start("warpperspective_gpu");
				cv::cuda::warpPerspective(imageFloat_gpu, imageWarped_gpu, map, imageWarped_gpu.size(), imageFlags, 0, Scalar(), stream[0]);
				timer.stop("warpperspective_gpu");
				timer.start("warpperspective_gpu");
				cv::cuda::warpPerspective(gradientX_gpu, gradientXWarped_gpu, map, gradientXWarped_gpu.size(), imageFlags, 0, Scalar(), stream[0]);
				timer.stop("warpperspective_gpu");
				timer.start("warpperspective_gpu");
				cv::cuda::warpPerspective(gradientY_gpu, gradientYWarped_gpu, map, gradientYWarped_gpu.size(), imageFlags, 0, Scalar(), stream[0]);
				timer.stop("warpperspective_gpu");
				timer.start("warpperspective_gpu");
				cv::cuda::warpPerspective(preMask_gpu, imageMask_gpu, map, imageMask_gpu.size(), maskFlags, 0, Scalar(), stream[0]);
				timer.stop("warpperspective_gpu");

				stream[0].waitForCompletion();  //stream sync
			}



			Scalar imgMean, imgStd, tmpMean, tmpStd;

			float sum_warp_val;
			timer.start("meanstd_gpu");
			Scalar sum_warp = cuda::sum(imageWarped_gpu);
			sum_warp_val = sum_warp.val[0];

			imgMean = sum_warp_val / (imageWarped_gpu.rows*imageWarped_gpu.cols);

			cv::cuda::GpuMat imageWarped_gpu_mid;
			cuda::subtract(imageWarped_gpu, imgMean, imageWarped_gpu_mid);
			Scalar sqrsum1 = cuda::sqrSum(imageWarped_gpu_mid);
			float squsum = sqrsum1.val[0];
			float x_std = squsum / (imageWarped_gpu.rows*imageWarped_gpu.cols);
			imgStd = sqrt(x_std);
			timer.stop("meanstd_gpu");

			//meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);
			float sum_warp_val2;
			timer.start("meanstd_gpu");
			Scalar sum_warp2 = cuda::sum(templateFloat_gpu);
			sum_warp_val2 = sum_warp2.val[0];

			tmpMean = sum_warp_val2 / (templateFloat_gpu.rows*templateFloat_gpu.cols);

			cv::cuda::GpuMat templateFloat_gpu_mid;
			cuda::subtract(templateFloat_gpu, tmpMean, templateFloat_gpu_mid);
			Scalar sqrsum2 = cuda::sqrSum(templateFloat_gpu_mid);
			float squsum2 = sqrsum2.val[0];
			float x_std2 = squsum2 / (templateFloat_gpu.rows*templateFloat_gpu.cols);
			tmpStd = sqrt(x_std2);
			timer.stop("meanstd_gpu");

			timer.start("subtract_gpu");
			cv::cuda::subtract(imageWarped_gpu, imgMean, imageWarped_gpu, imageMask_gpu);//zero-mean input
			timer.stop("subtract_gpu");
			templateZM_gpu = cv::cuda::GpuMat(templateZM_gpu.rows, templateZM_gpu.cols, templateZM_gpu.type(), double(0));

			timer.start("subtract_gpu");
			cv::cuda::subtract(templateFloat_gpu, tmpMean, templateZM_gpu, imageMask_gpu);//zero-mean template
			timer.stop("subtract_gpu");

			
			Xgrid_gpu.upload(Xgrid);
			Ygrid_gpu.upload(Ygrid);

			timer.start("count_gpu");
			const double tmpNorm = std::sqrt(cv::cuda::countNonZero(imageMask_gpu)*(tmpStd.val[0])*(tmpStd.val[0]));
			timer.stop("count_gpu");
			timer.start("count_gpu");
			const double imgNorm = std::sqrt(cv::cuda::countNonZero(imageMask_gpu)*(imgStd.val[0])*(imgStd.val[0]));
			timer.stop("count_gpu");

			jacobian_gpu.download(jacobian);
			if (motionType == MOTION_HOMOGRAPHY)

				image_jacobian_homo_ECC(gradientXWarped_gpu, gradientYWarped_gpu, Xgrid_gpu, Ygrid_gpu, map, jacobian);


			// calculate Hessian and its inverse

			project_onto_jacobian_ECC(jacobian, jacobian, hessian);

			hessianInv = hessian.inv();


			//const double correlation = templateZM.dot(imageWarped);

			cv::cuda::GpuMat correlation_gpu;

			timer.start("dot_gpu");
			cv::cuda::multiply(templateZM_gpu, imageWarped_gpu, correlation_gpu);
			Scalar corr = cuda::sum(correlation_gpu);
			const double correlation = corr[0];
			timer.stop("dot_gpu");

			templateZM_gpu.download(templateZM);
			imageWarped_gpu.download(imageWarped);


			// calculate enhanced correlation coefficiont (ECC)->rho
			last_rho = rho;
			rho = correlation / (imgNorm*tmpNorm);
			if (cvIsNaN(rho)) {
				CV_Error(Error::StsNoConv, "NaN encountered.");
			}

			// project images into jacobian

			project_onto_jacobian_ECC(jacobian, imageWarped, imageProjection);
			project_onto_jacobian_ECC(jacobian, templateZM, templateProjection);


			// calculate the parameter lambda to account for illumination variation
			imageProjectionHessian = hessianInv*imageProjection;
			const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
			const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
			if (lambda_d <= 0.0)
			{
				rho = -1;
				CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");

			}
			const double lambda = (lambda_n / lambda_d);

			// estimate the update step delta_p
			error = lambda*templateZM - imageWarped;
			project_onto_jacobian_ECC(jacobian, error, errorProjection);
			deltaP = hessianInv * errorProjection;

			// update warping matrix

			update_warping_matrix_ECC(map, deltaP, motionType);



		}


		// Storage for warped image.

		cv::cuda::GpuMat im2_aligned_gpu;


		if (motionType == MOTION_HOMOGRAPHY)
			// Use warpPerspective for Homography
			timer.start("warpperspective_gpu");
		cv::cuda::warpPerspective(im2_gpu, im2_aligned_gpu, map, im1_gpu.size(), INTER_LINEAR + WARP_INVERSE_MAP);
		timer.stop("warpperspective_gpu");
		//im1_gpu.download(im1);
		//im2_gpu.download(im2);
		im2_aligned_gpu.download(im2_aligned);

		// Show final result
		destroyAllWindows();
		imshow("Image 1", im1);
		imshow("Image 2", im2);
		imshow("Image 2 Aligned", im2_aligned);

		return rho;
	}

};


int main()
{
	cudaError_t status = cudaSetDevice(0);
	std::cout << status << std::endl;

	Mat sample_img = imread("1_new.png");

	cv::cuda::GpuMat color_in;
	cv::cuda::GpuMat color_out;
	color_in.upload(sample_img);
	cv::cuda::cvtColor(color_in, color_out, CV_BGR2GRAY);

	RegisterImagesCpu regimgcpu;
	RegisterImagesCuda regimgcuda(400, 200);//height, width
	//RegisterImagesCputoCuda regimgcputocuda;

	std::map< String, int>::iterator it;

	Mat  im1 = imread("1_new.png");
	Mat im2 = imread("2_new.png");

	//cuda_set_device(0);

	resize(im1, im1, Size(200, 400)); //width, height
	resize(im2, im2, Size(200, 400)); //width, height
	Mat inputMask(im1.size(), CV_8UC1, Scalar(1));
	Mat  im2_aligned;

	/*cout << "regimgcpu" << endl;
	for (int i = 0; i < 5; i++)
	{
		clock_start();
		int rho = regimgcpu.registerimg(im1, im2, inputMask, im2_aligned);
		clock_end("regimgcpu");
		//waitKey(0);
	}*/



	cout << "regimgcuda" << endl;
	for (int i = 0; i < 1; i++)
	{
		//clock_start();
		timer.start("regimgcuda");
		int rho = regimgcuda.registerimg(im1, im2, inputMask, im2_aligned);
		timer.stop("regimgcuda");
		//clock_end("regimgcuda");
		//waitKey(1);
	}

	timer.get_all_average_times();


	waitKey(0);
	return 0;
}
