#################Command to compile##########################################

g++ register_new.cpp -o register_new -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching -lopencv_videoio -lopencv_imgcodecs `pkg-config opencv --cflags --libs`

###################Command to run###########################################

./register_new
