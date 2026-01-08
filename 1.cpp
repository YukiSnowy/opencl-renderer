// g++ 1.cpp -g -lOpenCL

#include <iostream>
#include <vector>
#include <fstream>
#define CL_HPP_TARGET_OPENCL_VERSION 200 // 120 for old GPU
#include <CL/opencl.hpp>

std::string readKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

int main() {
std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        cl::Device device = devices[0];

        std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        const int width = 800;
        const int height = 600;
        size_t buffer_size = width * height * sizeof(cl_uint);
        cl::Buffer imgBuffer(context, CL_MEM_WRITE_ONLY, buffer_size);

        std::string kernelCode = readKernelSource("kernel.cl");
        
        cl::Program::Sources sources;
        sources.push_back({kernelCode.c_str(), kernelCode.length()});
        cl::Program program(context, sources);
        program.build({device});

        cl::Kernel kernel(program, "rasterize_interpolated_triangle");

        cl_float2 v0_pos = { (float)width / 2, 100.0f };
        cl_float2 v1_pos = { 100.0f, (float)height - 100.0f };
        cl_float2 v2_pos = { (float)width - 100.0f, (float)height - 100.0f };

        cl_uint v0_color = 0xFF0000;
        cl_uint v1_color = 0x00FF00;
        cl_uint v2_color = 0x0000FF;

        kernel.setArg(0, imgBuffer);
        kernel.setArg(1, width);
        kernel.setArg(2, height);
        kernel.setArg(3, v0_pos);
        kernel.setArg(4, v1_pos);
        kernel.setArg(5, v2_pos);
        kernel.setArg(6, v0_color);
        kernel.setArg(7, v1_color);
        kernel.setArg(8, v2_color);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height));
        queue.finish();

        std::vector<cl_uint> hostPixels(width * height);
        queue.enqueueReadBuffer(imgBuffer, CL_TRUE, 0, buffer_size, hostPixels.data());

        std::ofstream f("output.ppm");
        f << "P3\n" << width << " " << height << "\n255\n";
        for(auto p : hostPixels) {
            f << ((p >> 16) & 0xFF) << " " << ((p >> 8) & 0xFF) << " " << (p & 0xFF) << " ";
        }
        std::cout << "Done! Created output.ppm with interpolated triangle." << std::endl;
}