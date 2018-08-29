//// allocate page-locked memory
//CudaMem host_src_pl(768, 1024, CV_8UC1, CudaMem::ALLOC_PAGE_LOCKED);
//CudaMem host_dst_pl;
//
//// get Mat header for CudaMem (no data copy)
//Mat host_src = host_src_pl;
//
//// fill mat on CPU
//someCPUFunc(host_src);
//
//GpuMat gpu_src, gpu_dst;
//
//// create Stream object
//Stream stream;
//
//// next calls are non-blocking
//
//// first upload data from host
//stream.enqueueUpload(host_src_pl, gpu_src);
//// perform blur
//blur(gpu_src, gpu_dst, Size(5, 5), Point(-1, -1), stream);
//// download result back to host
//stream.enqueueDownload(gpu_dst, host_dst_pl);
//
//// call another CPU function in parallel with GPU
//anotherCPUFunc();
//
//// wait GPU for finish
//stream.waitForCompletion();
//
//// now you can use GPU results
//Mat host_dst = host_dst_pl;