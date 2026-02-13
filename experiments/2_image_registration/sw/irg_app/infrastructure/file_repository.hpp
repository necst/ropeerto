/******************************************
* MIT License
* 
* Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide Conficconi, Eleonora D'Arnese
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
*Copyright (c) [2019] [Davide Conficconi, Eleonora D'Arnese, Marco Domenico Santambrogio]
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
* credits goes to the author of this repo: https://github.com/mariusherzog/ImageRegistration
*
****************************************************************/

#ifndef IMAGE_FROM_FILE_HPP
#define IMAGE_FROM_FILE_HPP

#include <opencv2/opencv.hpp>

#include <string>
#include "../interfaces/image_repository.hpp"

/**
 * @brief The file_repository class implements the repository by loading the
 *        image from files.
 */


class file_repository : public Iimage_repository
{
   private:
      std::string path_reference;
      std::string path_floating;

   public:
      file_repository(std::string path_reference, std::string path_floating) :
         path_reference {path_reference},
         path_floating {path_floating}
      {
      }


      std::vector<cv::Mat> reference_image_3d(int volume)
      {  
         std::cout << "Going to read reference volume" << std::endl;
         std::vector<cv::Mat> output = std::vector<cv::Mat>(volume);
         // Itera su ogni immagine
         for (int i = 0; i < volume; i++) {
            std::string path = path_reference + "IM" + std::to_string(i) + ".png";
            cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
            output[i] = img;
         }
         return output;
      }


      std::vector<cv::Mat> floating_image_3d(int volume)
      {  
         std::cout << "Going to read floating volume" << std::endl;
         std::vector<cv::Mat> output = std::vector<cv::Mat>(volume);
         for (int i = 0; i < volume; i++) {
            std::string path = path_floating + "IM" + std::to_string(i) + ".png";
            cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
            output[i] = img;
         }
         
         return output;
      }


      cv::Mat reference_image() override
      {
         return cv::imread(path_reference, cv::IMREAD_GRAYSCALE);
      }

      cv::Mat floating_image() override
      {
         return cv::imread(path_floating, cv::IMREAD_GRAYSCALE);
      }
};

#endif // IMAGE_FROM_FILE_HPP
