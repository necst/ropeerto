/*
MIT License

Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide Conficconi, Eleonora D'Arnese

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "software_mi.hpp"

static double sw_registration_step_3d(uint8_t* input_ref, uint8_t* input_flt, uint8_t* output_flt, int n_couples, const int TX, const int TY, const float ANG,int depth, int padding){
    transform_volume(input_flt, output_flt, TX, TY, ANG, DIMENSION, (depth+padding),MODE_BILINEAR);
    
    double j_h[J_HISTO_ROWS][J_HISTO_COLS];
    for(int i=0;i<J_HISTO_ROWS;i++){
        for(int j=0;j<J_HISTO_COLS;j++){
            j_h[i][j]=0.0;
        }
    }

   const int N_COUPLES_TOTAL = depth + padding;

   for(int k = 0; k < N_COUPLES_TOTAL-padding; k++) {
      for(int i=0;i<DIMENSION;i++){
         for(int j=0;j<DIMENSION;j++){
               unsigned int a=input_ref[i * DIMENSION * (N_COUPLES_TOTAL) + j * (N_COUPLES_TOTAL) + k];
               unsigned int b=output_flt[i * DIMENSION * (N_COUPLES_TOTAL) + j * (N_COUPLES_TOTAL) + k];
               j_h[a][b]= (j_h[a][b])+1;
         }
      }
   }

   for (int i=0; i<J_HISTO_ROWS; i++) {
      for (int j=0; j<J_HISTO_COLS; j++) {
         j_h[i][j] = j_h[i][j]/((N_COUPLES_TOTAL-padding)*DIMENSION*DIMENSION); // per versal = dividere per n_couples anziché n_couples+padding
      }
   }

   float entropy = 0.0;
   for (int i=0; i<J_HISTO_ROWS; i++) {
      for (int j=0; j<J_HISTO_COLS; j++) {
         float v = j_h[j][i];
         if (v > 0.000000000000001) {
         entropy += v*log2(v);///log(2);
         }
      }
   }
   entropy *= -1;

   double href[J_HISTO_ROWS];
   for(int i=0;i<J_HISTO_ROWS;i++){
      href[i]=0.0;
   }

   for (int i=0; i<J_HISTO_ROWS; i++) {
      for (int j=0; j<J_HISTO_ROWS; j++) {
         href[i] += j_h[i][j];
      }
   }

   double hflt[J_HISTO_ROWS];
   for(int i=0;i<J_HISTO_ROWS;i++){
      hflt[i]=0.0;
   }

   for (int i=0; i<J_HISTO_ROWS; i++) {
      for (int j=0; j<J_HISTO_COLS; j++) {
         hflt[i] += j_h[j][i];
      }
   }


   double eref = 0.0;
   for (int i=0; i<J_HISTO_ROWS; i++) {
      if (href[i] > 0.000000000001) {
         eref += href[i] * log2(href[i]);///log(2);
      }
   }
   eref *= -1;

   double eflt = 0.0;
   for (int i=0; i<J_HISTO_ROWS; i++) {
      if (hflt[i] > 0.000000000001) {
         eflt += hflt[i] * log2(hflt[i]);///log(2);
      }
   }
   eflt =  eflt * (-1);

   double mutualinfo = eref + eflt - entropy;
   //delete[] output_flt;
   return mutualinfo;


}

static double sw_registration_step_3d(uint8_t* input_ref, uint8_t* input_flt,int n_couples, const int TX, const int TY, const float ANG,int depth, int padding){
    
    uint8_t* output_flt = new uint8_t[DIMENSION*DIMENSION * (depth+padding)];
    transform_volume(input_flt, output_flt, TX, TY, ANG, DIMENSION, (depth+padding),MODE_BILINEAR);
    // Calcolo della MI
    double j_h[J_HISTO_ROWS][J_HISTO_COLS];
    for(int i=0;i<J_HISTO_ROWS;i++){
        for(int j=0;j<J_HISTO_COLS;j++){
            j_h[i][j]=0.0;
        }
    }

   const int N_COUPLES_TOTAL = depth + padding;

   for(int k = 0; k < N_COUPLES_TOTAL-padding; k++) {
      for(int i=0;i<DIMENSION;i++){
         for(int j=0;j<DIMENSION;j++){
               unsigned int a=input_ref[i * DIMENSION * (N_COUPLES_TOTAL) + j * (N_COUPLES_TOTAL) + k];
               unsigned int b=output_flt[i * DIMENSION * (N_COUPLES_TOTAL) + j * (N_COUPLES_TOTAL) + k];
               j_h[a][b]= (j_h[a][b])+1;
         }
      }
   }

   //j_h[0][0] = j_h[0][0] - padding*DIMENSION*DIMENSION; // per versal = sottrarre i valori in j_h dovuti al padding

   for (int i=0; i<J_HISTO_ROWS; i++) {
      for (int j=0; j<J_HISTO_COLS; j++) {
         j_h[i][j] = j_h[i][j]/((N_COUPLES_TOTAL-padding)*DIMENSION*DIMENSION); // per versal = dividere per n_couples anziché n_couples+padding
      }
   }


   float entropy = 0.0;
   for (int i=0; i<J_HISTO_ROWS; i++) {
      for (int j=0; j<J_HISTO_COLS; j++) {
         float v = j_h[j][i];
         if (v > 0.000000000000001) {
         entropy += v*log2(v);///log(2);
         }
      }
   }
   entropy *= -1;

   double href[J_HISTO_ROWS];
   for(int i=0;i<J_HISTO_ROWS;i++){
      href[i]=0.0;
   }

   for (int i=0; i<J_HISTO_ROWS; i++) {
      for (int j=0; j<J_HISTO_ROWS; j++) {
         href[i] += j_h[i][j];
      }
   }

   double hflt[J_HISTO_ROWS];
   for(int i=0;i<J_HISTO_ROWS;i++){
      hflt[i]=0.0;
   }

   for (int i=0; i<J_HISTO_ROWS; i++) {
      for (int j=0; j<J_HISTO_COLS; j++) {
         hflt[i] += j_h[j][i];
      }
   }


   double eref = 0.0;
   for (int i=0; i<J_HISTO_ROWS; i++) {
      if (href[i] > 0.000000000001) {
         eref += href[i] * log2(href[i]);///log(2);
      }
   }
   eref *= -1;


   double eflt = 0.0;
   for (int i=0; i<J_HISTO_ROWS; i++) {
      if (hflt[i] > 0.000000000001) {
         eflt += hflt[i] * log2(hflt[i]);///log(2);
      }
   }
   eflt =  eflt * (-1);

   double mutualinfo = eref + eflt - entropy;
   delete[] output_flt;
   return mutualinfo;

}



static cv::Mat transform(cv::Mat image, double tx, double ty, double a11, double a12, double a21, double a22)
{
   cv::Mat trans_mat = (cv::Mat_<double>(2,3) << a11, a12, tx, a21, a22, ty);

   cv::Mat out = image.clone();
   warpAffine(image, out, trans_mat, image.size());
   return out;
}
