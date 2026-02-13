/******************************************
* MIT License
*
* Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide
Conficconi, Eleonora D'Arnese
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.

*MIT License
*
*Copyright (c) [2019] [Davide Conficconi, Eleonora D'Arnese, Marco Domenico
Santambrogio]
*
*Permission is hereby granted, free of charge, to any person obtaining a copy
*of this software and associated documentation files (the "Software"), to deal
*in the Software without restriction, including without limitation the rights
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*copies of the Software, and to permit persons to whom the Software is
*furnished to do so, subject to the following conditions:
*
*The above copyright notice and this permission notice shall be included in all
*copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*SOFTWARE.
*/
/***************************************************************
 *
 * credits goes to the author of this repo:
 *https://github.com/mariusherzog/ImageRegistration
 *
 ****************************************************************/
#ifndef DOMAIN_FUSION_HPP
#define DOMAIN_FUSION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "fusion_algorithms.hpp"
#include "register_algorithms.hpp"

/**
 * @brief available_fusion_algorithms
 * @return available strategies for fusion
 */
std::vector<std::string> available_fusion_algorithms() {
  return fusion_algorithms::available();
}

/**
 * @brief available_registration_algorithms
 * @return availabe strategies for registration
 */
std::vector<std::string> available_registration_algorithms() {
  return register_algorithms::available();
}

#ifndef HW_REG
/**
 * @brief fusion performs a fusion by registering the floating to the reference
 *        image and then perform the fusion
 * @param ref reference image
 * @param flt floating image
 * @param register_strategy method for registration
 * @param fusion_strategy method for fusion
 * @param n_couples number of couples
 * @param padding padding
 * @return execution time in seconds
 */

double fuse_images_3d(std::vector<cv::Mat> &ref, std::vector<cv::Mat> &flt,
                      std::string register_strategy,
                      std::string fusion_strategy, int n_couples, int padding,
                      int rangeX, int rangeY, float RangeAngZ,
                      uint8_t *registered_volume) {

  std::unique_ptr<fusion> fusion_algorithm =
      fusion_algorithms::pick(fusion_strategy);
  std::unique_ptr<registration> registration_algorithm =
      register_algorithms::pick(register_strategy);
  return registration_algorithm->register_images_3d(
      ref, flt, n_couples, padding, rangeX, rangeY, RangeAngZ,
      registered_volume);
}

#else
/**
 * @brief fusion performs a fusion by registering the floating to the reference
 *        image and then perform the fusion
 * @param ref reference image
 * @param flt floating image
 * @param register_strategy method for registration
 * @param fusion_strategy method for fusion
 * @param board reference to the board
 * @return execution time in seconds
 */

double fuse_images_3d(std::vector<cv::Mat> &ref, std::vector<cv::Mat> &flt,
                      std::string register_strategy,
                      std::string fusion_strategy,
                      HardwareAbstractionLayer &board, int rangeX, int rangeY,
                      float RangeAngZ) {
  using namespace cv;
  std::unique_ptr<fusion> fusion_algorithm =
      fusion_algorithms::pick(fusion_strategy);
  std::unique_ptr<registration> registration_algorithm =
      register_algorithms::pick(register_strategy);
  return registration_algorithm->register_images_3d(ref, flt, board, rangeX,
                                                    rangeY, RangeAngZ);
}
#endif

#endif // FUSION_HPP
