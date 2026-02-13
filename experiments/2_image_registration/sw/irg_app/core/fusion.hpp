
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

#ifndef FUSION_HPP
#define FUSION_HPP

#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


/**
 * @brief The fusion interface defines the signature of a fusion operation on
 *        two images (Mats).
 */
class fusion
{
   public:
      virtual cv::Mat fuse(cv::Mat a, cv::Mat b) = 0;
      virtual ~fusion() = 0;
};

fusion::~fusion()
{
}

/**
 * @brief The alphablend strategy performs a fusion by alpha blending both
 *        images
 */
class alphablend : public fusion
{
   public:
      alphablend(double alpha) :
         alpha {alpha}
      {
      }

      cv::Mat fuse(cv::Mat ref, cv::Mat flt) override
      {
         assert(abs(alpha) < 1.0);
         
         cv::Mat color(flt.cols, flt.rows, CV_8UC3);
         cv::cvtColor(flt, color, cv::COLOR_GRAY2BGR);
         cv::Mat channel[3];
         split(color, channel);
         channel[1] = cv::Mat::zeros(flt.rows, flt.cols, CV_8UC1);
         merge(channel, 3, color);

         cv::cvtColor(ref, ref, cv::COLOR_GRAY2BGR);

         const double beta = 1-alpha;
         cv::Mat dst = ref.clone();
         cv::addWeighted(ref, alpha, color, beta, 0.0, dst);
         return dst;
      }

   private:
      const double alpha;
};

#endif // FUSION_HPP