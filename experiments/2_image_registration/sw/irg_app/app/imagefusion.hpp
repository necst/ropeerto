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
#ifndef IMAGEFUSION_HPP
#define IMAGEFUSION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#ifndef HW_REG
#include "../core/fusion_services.hpp"
#include "../infrastructure/file_repository.hpp"

class imagefusion {
public:
#ifndef HW_REG
  /**
   * @brief perform_fusion_from_files is an application service which
   *        performs fusion of two images and the given register / fusion
   *        algorithms
   * @param path_reference_image path to the reference image
   * @param path_floating_image path to the floating image
   * @param register_strategy name of the register strategy
   * @param fusion_strategy name of the fusion strategy
   * @param n_couples number of couples
   * @param padding padding
   * @param registered_image reference to the registered image
   * @return execution time in seconds
   */
  static double perform_fusion_from_files_3d(
      std::vector<cv::Mat> &reference_image,
      std::vector<cv::Mat> &floating_image, std::string register_strategy,
      std::string fusion_strategy, int n_couples, int padding, int rangeX,
      int rangeY, float rangeAngZ, uint8_t *registered_volume)

  {

    return fuse_images_3d(reference_image, floating_image, register_strategy,
                          fusion_strategy, n_couples, padding, rangeX, rangeY,
                          rangeAngZ, registered_volume);
  }

#else
  /**
   * @brief perform_fusion_from_files is an application service which
   *        performs fusion of two images and the given register / fusion
   *        algorithms
   * @param path_reference_image path to the reference image
   * @param path_floating_image path to the floating image
   * @param register_strategy name of the register strategy
   * @param fusion_strategy name of the fusion strategy
   * @param board reference to the board
   * @return execution time in seconds
   */
  static double perform_fusion_from_files_3d(
      std::vector<cv::Mat> &reference_image,
      std::vector<cv::Mat> &floating_image, std::string register_strategy,
      std::string fusion_strategy, HardwareAbstractionLayer &board, int rangeX,
      int rangeY, float rangeAngZ) {

    return fuse_images_3d(reference_image, floating_image, register_strategy,
                          fusion_strategy, board, rangeX, rangeY, rangeAngZ);
  }
#endif

  /**
   * @brief fusion_strategies queries availabe fusion strategies
   * @return a list of availabe fusion strategies
   */
  static std::vector<std::string> fusion_strategies() {
    return available_fusion_algorithms();
  }

  /**
   * @brief register_strategies queries availabe register strategies
   * @return a list of availabe register strategies
   */
  static std::vector<std::string> register_strategies() {
    return available_registration_algorithms();
  }
};

#endif // IMAGEFUSION_HPP
