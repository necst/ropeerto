/******************************************
* MIT License
*
* Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide Conficconi, Eleonora
D'Arnese
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
#ifndef FUSION_ALGORITHMS_HPP
#define FUSION_ALGORITHMS_HPP

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>

#include "fusion.hpp"

template <typename T, typename... Args>

static std::unique_ptr<T> custom_make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/**
 * @brief The fusion_algorithms class is a facade which facilitates easy access
 *        to all available fusion strategies. Each strategies is uniquely
 *        identified by its name and can be accessed with it.
 */
class fusion_algorithms {
   private:
    static const std::vector<std::string> algorithms;

   public:
    /**
     * @brief pick returns the fusion algorithm for the given name
     * @param name identifying name / key of the fusion strategy
     * @return instance of the fusion strategy
     */
    static std::unique_ptr<fusion> pick(std::string name) {
        if (std::find(algorithms.begin(), algorithms.end(), name) == algorithms.end()) {
            name = "alphablend";
        }

        if (name == "alphablend") {
            return custom_make_unique<alphablend>(0.5);
        }
        return nullptr;
    }

    /**
     * @brief available returns a list of the names of all available fusion
     *        algorithms.
     * @return list of names of fusion algorithms
     */
    static std::vector<std::string> available() {
        return algorithms;
    }
};

/**
 * @todo maybe replace by enum
 */
const std::vector<std::string> fusion_algorithms::algorithms{"alphablend"};

#endif  // FUSION_ALGORITHMS_HPP
