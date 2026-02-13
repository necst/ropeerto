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
* factory method for the registration algorithms
* credits goes also to the author of this repo: https://github.com/mariusherzog/ImageRegistration
*
****************************************************************/
#ifndef REGISTER_ALGORITHMS_HPP
#define REGISTER_ALGORITHMS_HPP

#include <string>
#include <algorithm>
#include <memory>
#include <cassert>

#include "register.hpp"

/**
 * @brief The fusion_algorithms class is a facade which facilitates easy access
 *        to all available fusion strategies. Each strategies is uniquely
 *        identified by its name and can be accessed with it.
 */
class register_algorithms
{
   private:
      static const std::vector<std::string> algorithms;

   public:
      /**
       * @brief pick returns the fusion algorithm for the given name
       * @param name identifying name / key of the fusion strategy
       * @return instance of the fusion strategy
       */
      static std::unique_ptr<registration> pick(std::string name)
      {
         if (std::find(algorithms.begin(), algorithms.end(), name) == algorithms.end())
         {
            name = "mutualinformation";
         }

         if (name == "mutualinformation")
         {
            return std::make_unique<mutualinformation>();
         }
         
         return nullptr;
      }

      /**
       * @brief available returns a list of the names of all available
       *        registration algorithms.
       * @return list of names of registration algorithms
       */
      static std::vector<std::string> available()
      {
         return algorithms;
      }
};

/**
 * @todo maybe replace by enum
 */
const std::vector<std::string> register_algorithms::algorithms
{
   "mutualinformation",
   "identity"
};

#endif // REGISTER_ALGORITHMS_HPP
