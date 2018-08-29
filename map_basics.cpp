#include <iostream>
#include "TimeProfiler_.h"
#include "opencv2/opencv.hpp"
#include <sstream>
using namespace cv;
#include <chrono>
//#include <map>



int main(int argc, char* argv[])
{

	std::map< String, int> dic;
	dic["one"] = 1;
	dic["two"] = 3;

	std::map< String, int>::iterator it;

	std::cout << dic["two"] << std::endl;
	//std::cout << dic["three"] << std::endl;
	it = dic.find("three");
	if (it != dic.end())
		std::cout << dic.find("three")->second << std::endl;
	else
		std::cout << "three- not a valid key" << std::endl;

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
}