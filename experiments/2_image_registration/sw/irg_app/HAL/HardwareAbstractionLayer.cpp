// HardwareAbstractionLayer.cpp
#include "HardwareAbstractionLayer.h"
#include <iostream>

HardwareAbstractionLayer::HardwareAbstractionLayer(
    const device &device, int resolution_, int depth_,
    RigidWarpXYPlane &transformer_)
#ifdef COYOTE_MODE
    : coyote_thread(device.vfpga_index, getpid(), device.device_index),
      resolution(resolution_), depth(depth_), transformer(transformer_),
      p2p_mode(device.p2p_mode)
#else
    : resolution(resolution_), depth(depth_), transformer(transformer_)
#endif
{
  printGPUCapabilities_HIP();

  // Compute buffer size (voxels * sizeof)
  size_t num_voxels = resolution * resolution * depth;
  uint32_t allocSize = num_voxels * sizeof(uint8_t);

#ifdef COYOTE_MODE

  float_cpu = new uint8_t[num_voxels];

  if(p2p_mode) {
    ptr_flt = (uint8_t *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::GPU,
       allocSize, false, (uint32_t) device.gpu_index});
    ptr_out = (uint8_t *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::GPU,
       allocSize, false, (uint32_t) device.gpu_index});
  } else {
    ptr_flt = (uint8_t *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::HPF,
       allocSize});
    ptr_out = (uint8_t *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::HPF,
       allocSize});
  }
  
  ptr_ref =
      (uint8_t *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, allocSize});
  
  mutual_info = (float *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::HPF, 1 * sizeof(float)});
  n_couples_mem = (uint64_t *)coyote_thread.getMem(
      {coyote::CoyoteAllocType::HPF, sizeof(uint64_t)});
  if (!ptr_flt || !ptr_ref || !ptr_out || !mutual_info || !n_couples_mem) {
    throw std::runtime_error(
        "Could not allocate memory for vectors, exiting...");
  }

  if (p2p_mode) {

    std::cout << "Warming up HIP kernel...\n";
    for (int i = 0; i < 10; i++) {
      transformer.moveToGPU(ptr_flt, float_cpu, resolution, depth);
      transformer.run_external(ptr_flt, ptr_out, 0, 0, 0, resolution, depth);
      transformer.moveFromGPU(float_cpu, ptr_out, resolution, depth);
    }
    std::cout << "Finished warming up HIP kernel" << std::endl;
  }

  // std::cout << "Buffers allocated" << std::endl;

#else

  // Open device
  device = xrt::device(device.device_index);
  // std::cout << "Device opened" << std::endl;
  //  Program it
  auto uuid = device.load_xclbin(device.xclbin_path);
  krnl = xrt::kernel(device, uuid, device.kernel_name);
  // std::cout << "Kernel loaded" << std::endl;

  // Allocate three BOs (ref,flt,out)
  bo_flt = xrt::bo(device, allocSize, krnl.group_id(0));
  bo_ref = xrt::bo(device, allocSize, krnl.group_id(1));
  bo_out = xrt::bo(device, sizeof(float), krnl.group_id(2));
  ////std::cout << "BOs allocated" << std::endl;

  // std::cout << "Allocating " << num_voxels << " voxels" << std::endl;
  ptr_flt = new uint8_t[num_voxels];
  ptr_ref = new uint8_t[num_voxels];
  ptr_out = new uint8_t[num_voxels];

  runner = xrt::run(krnl);
  // Set kernel arguments
  runner.set_arg(0, bo_flt);
  runner.set_arg(1, bo_ref);
  runner.set_arg(2, bo_out);
  runner.set_arg(3, depth);
  runner.set_arg(4, 0);

#endif
  ////std::cout << "HAL created" << std::endl;
}

HardwareAbstractionLayer::~HardwareAbstractionLayer() {

#ifdef COYOTE_MODE

  // std::cout << "Destroying Coyote thread" << std::endl;
  coyote_thread.userUnmap((void *)ptr_flt);
  coyote_thread.userUnmap((void *)ptr_ref);
  coyote_thread.userUnmap((void *)ptr_out);
  coyote_thread.userUnmap((void *)mutual_info);
  coyote_thread.userUnmap((void *)n_couples_mem);

  delete float_cpu;

#else

  // xrt::bo and xrt::kernel clean up automatically
  delete ptr_flt;
  delete ptr_ref;
  delete ptr_out;

#endif
  // std::cout << "HAL destroyed" << std::endl;
}

void HardwareAbstractionLayer::load_ref(const std::string &folder) {
  // fill host buffer, then push to device
  // std::cout << "Loading reference volume from folder: " << folder <<
  // std::endl;
  read_volume_from_folder(ptr_ref, resolution, depth, folder);

#ifndef COYOTE_MODE

  // std::cout << "Writing reference volume to device" << std::endl;
  bo_ref.write(ptr_ref);
  bo_ref.sync(XCL_BO_SYNC_BO_TO_DEVICE);

#endif
  // std::cout << "Reference volume loaded" << std::endl;
}

void HardwareAbstractionLayer::load_flt(const std::string &folder) {
  // fill host buffer, then push to device
  // std::cout << "Loading filter volume from folder: " << folder << std::endl;
#ifdef COYOTE_MODE
  if (p2p_mode) {
    read_volume_from_folder(float_cpu, resolution, depth, folder);
    transformer.moveToGPU(ptr_flt, float_cpu, resolution, depth);
  } else {
    read_volume_from_folder(ptr_flt, resolution, depth, folder);
    transformer.transferToGPU(ptr_flt, resolution, depth);
  }
#else
  read_volume_from_folder(ptr_flt, resolution, depth, folder);

  bo_flt.write(ptr_flt);
  bo_flt.sync(XCL_BO_SYNC_BO_TO_DEVICE);

#endif
  // std::cout << "Floating volume loaded" << std::endl;
}

float HardwareAbstractionLayer::compute_mi(uint8_t *curr_ptr_float) {

  // std::cout << "Computing mutual information" << std::endl;

#ifdef COYOTE_MODE

  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();

  uint32_t local_write_count =
      coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE);
  uint32_t local_read_count =
      coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_READ);

  //coyote_thread->clearCompleted();

  // std::cout << "Writing filter volume to Coyote thread" << std::endl;
  size_t num_voxels = resolution * resolution * depth;
  uint32_t allocSize = num_voxels * sizeof(uint8_t);
  coyote::localSg sg_out;
  memset(&sg_out, 0, sizeof(coyote::localSg));
  sg_out = {
      .addr = curr_ptr_float, .len = allocSize, .dest = 0};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg_out);
  // std::cout << "Floating volume written to Coyote thread" << std::endl;

  coyote::localSg sg_ref;
  memset(&sg_ref, 0, sizeof(coyote::localSg));
  sg_ref = {.addr = ptr_ref, .len = allocSize, .dest = 1};
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg_ref);
  // std::cout << "Reference volume written to Coyote thread" << std::endl;

  coyote::localSg sg_n_couples, sg_mutual_info;

  memset(&sg_n_couples, 0, sizeof(coyote::localSg));
  memset(&sg_mutual_info, 0, sizeof(coyote::localSg));

  // Set the number of couples in the kernel

  sg_n_couples = {
      .addr = n_couples_mem, .len = sizeof(uint64_t), .dest = 2};
  sg_mutual_info = {
      .addr = mutual_info, .len = sizeof(float), .dest = 0};

  n_couples_mem[0] = (uint64_t)depth;

  // std::cout << "Number of couples: " << n_couples_mem[0] << std::endl;

  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg_n_couples);

  // sleep for 3 seconds
  //std::this_thread::sleep_for(std::chrono::seconds(3));

  // std::cout << "Number of couples set in Coyote thread" << std::endl;

  //std::cout << "Local read count: " << local_read_count
  //          << ", Local write count: " << local_write_count << std::endl;

  //std::cout << "CheckCompleted LOCAL_READ: "
  //          << coyote_thread->checkCompleted(coyote::CoyoteOper::LOCAL_READ)
  //          << std::endl;
  /*while (coyote_thread->checkCompleted(coyote::CoyoteOper::LOCAL_READ) !=
       3) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(50));
       }
  */

  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  //std::cout << "FPGA Read in " << elapsed_seconds.count() << std::endl;
  // Run the kernel
  coyote_thread.setCSR(static_cast<uint64_t>(0x1), static_cast<uint32_t>(0));

  // std::cout << "Control register set in Coyote thread" << std::endl;

  std::chrono::high_resolution_clock::time_point start_run =
      std::chrono::high_resolution_clock::now();
  // Retrieving mutual information
  coyote_thread.invoke(coyote::CoyoteOper::LOCAL_WRITE, sg_mutual_info);

  while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) <= local_write_count) {};

  //std::cout << "CheckCompleted LOCAL_READ: "
  //          << coyote_thread->checkCompleted(coyote::CoyoteOper::LOCAL_READ)
  //          << std::endl;

  std::chrono::high_resolution_clock::time_point end_run =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds_run = end_run - start_run;
  //std::cout << "MI kernel + FPGA write in " << elapsed_seconds_run.count()
  //          << std::endl;

  // std::cout << "Mutual information computed in Coyote thread" << std::endl;

  return mutual_info[0];

#else

  bo_flt.write(curr_ptr_float);
  bo_flt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // std::cout << "Written Float" << std::endl;
  runner.start();
  runner.wait();
  // std::cout << "Kernel execution finished" << std::endl;
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  float mi;
  bo_out.read(&mi);
  return mi;

#endif
}

float HardwareAbstractionLayer::run_reg_step(float tx, float ty, float ang) {

  // std::cout << "Running registration step" << std::endl;
  // measure time precisely
  // std::chrono::high_resolution_clock::time_point start =
  //    std::chrono::high_resolution_clock::now();

  // If we are in P2P mode, we need to transfer the filter volume to the GPU
  // before running the kernel
#ifdef COYOTE_MODE
  bool complete = !p2p_mode;
  // std::cout << "complete: " << complete << "p2p_mode: " << p2p_mode
  // << std::endl;
  transform_volume(tx, ty, ang, complete);

  counter++;
#else
  transform_volume(tx, ty, ang);
#endif
  // Transfer the output to the device
  // std::cout << "Computing MI" << std::endl;

  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
  float mi = compute_mi(ptr_out);
  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  //std::cout << "MI computed in " << elapsed_seconds.count() << " seconds"
  //          << std::endl;

  // std::chrono::high_resolution_clock::time_point end =
  //     std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_seconds = end - start;
  // std::cout << "Registration step completed in " << elapsed_seconds.count()
  //           << std::endl;

  // std::cout << "Mutual Information: " << mi << std::endl;
  return mi;
}

void HardwareAbstractionLayer::transform_volume(float tx, float ty, float ang,
                                                bool complete) {
// std::cout << "Transforming volume " << std::endl;
#ifdef COYOTE_MODE
  if (p2p_mode) {
    /*
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    transformer.moveToGPU(ptr_flt, float_cpu, resolution, depth);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Transfer to GPU took " << elapsed_seconds.count() << "seconds"
              << std::endl;
    */
    std::chrono::high_resolution_clock::time_point start_run =
        std::chrono::high_resolution_clock::now();
    double t = transformer.run_external(ptr_flt, ptr_out, tx, ty, ang,
                                        resolution, depth);
    std::chrono::high_resolution_clock::time_point end_run =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_run = end_run - start_run;
    //std::cout << "Kernel execution took " << elapsed_seconds_run.count()
    //          << " seconds" << std::endl;
    if (complete) {
      transformer.moveFromGPU(float_cpu, ptr_out, resolution, depth);
    }
  } else {
    // measure time precisely
    /*
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    transformer.transferToGPU(ptr_flt, resolution, depth);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Transfer to GPU took " << elapsed_seconds.count() << "seconds"
              << std::endl;
    */
    std::chrono::high_resolution_clock::time_point start_run =
        std::chrono::high_resolution_clock::now();
    int ret_val = transformer.run(tx, ty, ang);
    std::chrono::high_resolution_clock::time_point end_run =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_run = end_run - start_run;
    //std::cout << "Kernel execution took " << elapsed_seconds_run.count()
    //          << " seconds" << std::endl;
    if (complete) {
      std::chrono::high_resolution_clock::time_point start_transfer =
          std::chrono::high_resolution_clock::now();
      transformer.transferFromGPU(ptr_out);
      std::chrono::high_resolution_clock::time_point end_transfer =
          std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds_transfer =
          end_transfer - start_transfer;
      //std::cout << "Transfer from GPU took " << elapsed_seconds_transfer.count()
      //          << " seconds" << std::endl;
    }
  }

#else

  transformer.transferToGPU(ptr_flt, resolution, depth);
  int ret_val = transformer.run(tx, ty, ang);
  if (complete) {
    transformer.transferFromGPU(ptr_out);
  }

#endif

  // std::cout << "Transformed volume " << std::endl;
}
