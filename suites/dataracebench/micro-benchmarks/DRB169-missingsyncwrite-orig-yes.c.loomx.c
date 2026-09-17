/*
!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!
!!! Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
!!! and DataRaceBench project contributors. See the DataRaceBench/COPYRIGHT file for details.
!!!
!!! SPDX-License-Identifier: (BSD-3-Clause)
!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!
*/
/* This kernel imitates the nature of a program from the NAS Parallel Benchmarks 3.0 MG suit.
 * Due to missing construct to write r1[k]@38:9 synchronously, there is a Data Race.
 * Data Race Pair, r1[k]@38:9:W vs. r1[k]@38:9:W
 * */
#include <stdio.h>
#include <omp.h>
#define N 8

int main()
{
  int i;
  int j;
  int k;
  double r1[8];
  double r[8][8][8];
  for (i = 0; i < 8; i++) {
    for (j = 0; j < 8; j++) {
      for (k = 0; k < 8; k++) {
        r[i][j][k] = i;
      }
    }
  }
  
#pragma omp parallel for default(shared) private(j,k)
  for (i = 1; i < 8 - 1; i++) {
    for (j = 1; j < 8 - 1; j++) {
      for (k = 0; k < 8; k++) {
        r1[k] = r[i][j - 1][k] + r[i][j + 1][k] + r[i - 1][j][k] + r[i + 1][j][k];
      }
    }
  }
  for (k = 0; k < 8; k++) 
    printf("%f ",r1[k]);
  printf("\n");
  return 0;
}
