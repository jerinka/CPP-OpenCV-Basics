#ifndef TIMEPROFILER__H
#define TIMEPROFILER__H
/*Bottom, u can see sample usage of header file*/
#include <string>
#include <iostream>
#include <chrono>
#include<map>

class TimerElement
{
private:
	std::string name;
	int call_count;
	float total_time;
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point stop_time;

public:
	TimerElement()
	{}

	TimerElement(std::string name1)
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
		float t = std::chrono::duration_cast<std::chrono::milliseconds>(time_interval).count();
		std::cout << "Single run time for : " << name << " = " << t << " milli sec" << std::endl;
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
	std::map< std::string, TimerElement> timer_dic;
	std::map< std::string, TimerElement> it;

	void createNewTimer(std::string name)
	{
		TimerElement temp(name);

		timer_dic.insert(std::make_pair(name, temp));

	}

public:
	TimeProfiler()
	{
	}

	void start_timer(std::string name)
	{
		createNewTimer(name);
		timer_dic[name].start();
	}

	void stop_timer(std::string name)
	{
		if (timer_dic.count(name) == 1)
			timer_dic[name].stop();
		else
			std::cout << "timer: " << name << "doesn't exist" << std::endl;
	}

	float get_total_time(std::string name)
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

	int get_call_count(std::string name)
	{
		if (timer_dic.count(name) == 1)
		{
			int call_count = timer_dic[name].get_call_count();
			std::cout << "Call count of timer : " << name << " = " << call_count << " sec" << std::endl;
			return call_count;
		}
		else
		{
			std::cout << "timer: " << name << "doesn't exist" << std::endl;
			return 0;
		}
	}


	float get_average_time(std::string name)
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
//  timer.get_call_count("imread");
//  timer.get_average_time("imread");
//}
//#################################################################################################################


