#ifndef TIMEPROFILER__H
#define TIMEPROFILER__H
/*Bottom, u can see sample usage of header file*/
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
using namespace cv;
#include <chrono>


class TimerElement
{
private:
	String name;
	int call_count;
	float total_time;
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point stop_time;

public:
	TimerElement()
	{}

	TimerElement(String name1)
	{
		name = name1;
		total_time = 0;
		call_count = 0;
	}


	void start()
	{
		start_time = std::chrono::steady_clock::now();
		
	}

	void stop()
	{
		stop_time = std::chrono::steady_clock::now();
		std::chrono::duration<float> time_interval = stop_time - start_time;
		float t = time_interval.count();
		std::cout << "Single run time for : " << name << " = " << t << " sec" << std::endl;
		total_time += t;
		call_count++; //increment after every call
	}

	float get_total_time()
	{
		return total_time;
	}
	
	int get_call_count()
	{
		return call_count;
	}


};


class TimeProfiler
{
	std::map< String, TimerElement> timer_dic;
	std::map< String, TimerElement> it;

	void createNewTimer(String name)
	{
		TimerElement temp(name);

		//it = timer_dic.find(name);
		//if (it != timer_dic.end())
		/*if (timer_dic.count(name) == 0)
		{
		std::cout << "creating timer:  " << name << std::endl;
		TimerElement temp(name);
		timer_dic.insert(std::pair<String, TimerElement>(name, temp));
		}*/

		//timer_dic[name] = temp;
		timer_dic.insert(std::make_pair(name, temp));

		//TimerElement temp(name);

		//typedef std::map<String, TimerElement> map_type;

		//std::pair<typename map_type::iterator, bool> p
		//	= timer_dic.insert(std::pair<String const &, TimerElement const &>(name, temp));

		//if (!p.second) p.first->second = temp;

	}

public:
	TimeProfiler()
	{
	}

	void start_timer(String name)
	{
		createNewTimer(name);
		timer_dic[name].start();
	}

	void stop_timer(String name)
	{
		if (timer_dic.count(name) == 1)
			timer_dic[name].stop();
		else
			std::cout << "timer: " << name << "doesn't exist" << std::endl;
	}

	float get_total_time(String name)
	{
		if (timer_dic.count(name) == 1)
		{
			float t = timer_dic[name].get_total_time();
			std::cout << "Total  run time for : " << name << " = " << t << " sec" << std::endl;
			return (t);
		}
		else
			std::cout << "timer: " << name << "doesn't exist" << std::endl;
		    return (NAN);
	}


	float get_average_time(String name)
	{
		if (timer_dic.count(name) == 1)
		{
			int call_count = timer_dic[name].get_call_count();
			float total_time = timer_dic[name].get_total_time();
			float average_time = total_time / call_count;
			if (call_count > 0)
			{
				std::cout << "Average  run time for : " << name << " = " << average_time << " sec" << std::endl;
				return average_time;
			}
			else
			{
				std::cout << "Timer not was called even once" << std::endl;
				return(NAN);
			}
		}
		else
			std::cout << "timer: " << name << "doesn't exist" << std::endl;
		return (NAN);
	}

};
#endif // !TIMEPROFILER__H


////###########################################    example   ######################################################

//int main()
//{
//	TimeProfiler timer;
//
//	timer.start_timer("imread");
//	Mat im1 = imread("filename.png");
//	timer.stop_timer("imread");
//
//	timer.start_timer("imread");
//	Mat im2 = imread("1_new.png");
//	timer.stop_timer("imread");
//
//
//	timer.get_total_time("imread");
//  timer.get_average_time("imread");
//}
//#################################################################################################################


