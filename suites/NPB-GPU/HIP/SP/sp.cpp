/* 
 * ------------------------------------------------------------------------------
 *
 * MIT License
 *
 * Copyright (c) 2021 Parallel Applications Modelling Group - GMAP
 *      GMAP website: https://gmap.pucrs.br
 *
 * Pontifical Catholic University of Rio Grande do Sul (PUCRS)
 * Av. Ipiranga, 6681, Porto Alegre - Brazil, 90619-900
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
 *
 * ------------------------------------------------------------------------------
 *
 * The original NPB 3.4 version was written in Fortran and belongs to:
 *      http://www.nas.nasa.gov/Software/NPB/
 *
 * Authors of the Fortran code:
 *      R. Van der Wijngaart 
 *      W. Saphir 
 *      H. Jin
 *
 * ------------------------------------------------------------------------------
 *
 * The serial C++ version is a translation of the original NPB 3.4
 * Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER
 *
 * Authors of the C++ code:
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Araujo <hexenoften@gmail.com>
 *      Júnior Löff <loffjh@gmail.com>
 *
 * ------------------------------------------------------------------------------
 *
 * The hip version is a parallel implementation of the serial C++ version
 * hip version: https://github.com/GMAP/NPB-GPU/tree/master/hip
 *
 * Authors of the hip code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 */

#include <hip/hip_runtime.h>
#include "../common/npb.hpp"
#include "npbparams.hpp"

#define IMAX (PROBLEM_SIZE)
#define JMAX (PROBLEM_SIZE)
#define KMAX (PROBLEM_SIZE)
#define IMAXP (IMAX/2*2)
#define JMAXP (JMAX/2*2)
#define PROFILING_TOTAL_TIME (0)

#define PROFILING_ADD (1)
#define PROFILING_COMPUTE_RHS_1 (2)
#define PROFILING_COMPUTE_RHS_2 (3)
#define PROFILING_ERROR_NORM_1 (4)
#define PROFILING_ERROR_NORM_2 (5)
#define PROFILING_EXACT_RHS_1 (6)
#define PROFILING_EXACT_RHS_2 (7)
#define PROFILING_EXACT_RHS_3 (8)
#define PROFILING_EXACT_RHS_4 (9)
#define PROFILING_INITIALIZE (10)
#define PROFILING_RHS_NORM_1 (11)
#define PROFILING_RHS_NORM_2 (12)
#define PROFILING_TXINVR (13)
#define PROFILING_X_SOLVE (14)
#define PROFILING_Y_SOLVE (15)
#define PROFILING_Z_SOLVE (16)

/* gpu linear pattern */
#define u(m,i,j,k) u[(i)+nx*((j)+ny*((k)+nz*(m)))]
#define forcing(m,i,j,k) forcing[(i)+nx*((j)+ny*((k)+nz*(m)))]
#define rhs(m,i,j,k) rhs[m+(i)*5+(j)*5*nx+(k)*5*nx*ny]
#define rho_i(i,j,k) rho_i[i+(j)*nx+(k)*nx*ny]
#define us(i,j,k) us[i+(j)*nx+(k)*nx*ny]
#define vs(i,j,k) vs[i+(j)*nx+(k)*nx*ny]
#define ws(i,j,k) ws[i+(j)*nx+(k)*nx*ny]
#define square(i,j,k) square[i+(j)*nx+(k)*nx*ny]
#define qs(i,j,k) qs[i+(j)*nx+(k)*nx*ny]
#define speed(i,j,k) speed[i+(j)*nx+(k)*nx*ny]

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static double u_host[KMAX][JMAXP+1][IMAXP+1][5];
static double us_host[KMAX][JMAXP+1][IMAXP+1];
static double vs_host[KMAX][JMAXP+1][IMAXP+1];
static double ws_host[KMAX][JMAXP+1][IMAXP+1];
static double qs_host[KMAX][JMAXP+1][IMAXP+1];
static double rho_i_host[KMAX][JMAXP+1][IMAXP+1];
static double speed_host[KMAX][JMAXP+1][IMAXP+1];
static double square_host[KMAX][JMAXP+1][IMAXP+1];
static double rhs_host[KMAX][JMAXP+1][IMAXP+1][5];
static double forcing_host[KMAX][JMAXP+1][IMAXP+1][5];
static double cv_host[PROBLEM_SIZE];
static double rhon_host[PROBLEM_SIZE];
static double rhos_host[PROBLEM_SIZE];
static double rhoq_host[PROBLEM_SIZE];
static double cuf_host[PROBLEM_SIZE];
static double q_host[PROBLEM_SIZE];
static double ue_host[5][PROBLEM_SIZE];
static double buf_host[5][PROBLEM_SIZE];
static double lhs_host[IMAXP+1][IMAXP+1][5];
static double lhsp_host[IMAXP+1][IMAXP+1][5];
static double lhsm_host[IMAXP+1][IMAXP+1][5];
#else
static double (*u_host)[JMAXP+1][IMAXP+1][5]=(double(*)[JMAXP+1][IMAXP+1][5])malloc(sizeof(double)*((KMAX)*(JMAXP+1)*(IMAXP+1)*(5)));
static double (*us_host)[JMAXP+1][IMAXP+1]=(double(*)[JMAXP+1][IMAXP+1])malloc(sizeof(double)*((KMAX)*(JMAXP+1)*(IMAXP+1)));
static double (*vs_host)[JMAXP+1][IMAXP+1]=(double(*)[JMAXP+1][IMAXP+1])malloc(sizeof(double)*((KMAX)*(JMAXP+1)*(IMAXP+1)));
static double (*ws_host)[JMAXP+1][IMAXP+1]=(double(*)[JMAXP+1][IMAXP+1])malloc(sizeof(double)*((KMAX)*(JMAXP+1)*(IMAXP+1)));
static double (*qs_host)[JMAXP+1][IMAXP+1]=(double(*)[JMAXP+1][IMAXP+1])malloc(sizeof(double)*((KMAX)*(JMAXP+1)*(IMAXP+1)));
static double (*rho_i_host)[JMAXP+1][IMAXP+1]=(double(*)[JMAXP+1][IMAXP+1])malloc(sizeof(double)*((KMAX)*(JMAXP+1)*(IMAXP+1)));
static double (*speed_host)[JMAXP+1][IMAXP+1]=(double(*)[JMAXP+1][IMAXP+1])malloc(sizeof(double)*((KMAX)*(JMAXP+1)*(IMAXP+1)));
static double (*square_host)[JMAXP+1][IMAXP+1]=(double(*)[JMAXP+1][IMAXP+1])malloc(sizeof(double)*((KMAX)*(JMAXP+1)*(IMAXP+1)));
static double (*rhs_host)[JMAXP+1][IMAXP+1][5]=(double(*)[JMAXP+1][IMAXP+1][5])malloc(sizeof(double)*((KMAX)*(JMAXP+1)*(IMAXP+1)*(5)));
static double (*forcing_host)[JMAXP+1][IMAXP+1][5]=(double(*)[JMAXP+1][IMAXP+1][5])malloc(sizeof(double)*((KMAX)*(JMAXP+1)*(IMAXP+1)*(5)));
static double (*cv_host)=(double*)malloc(sizeof(double)*(PROBLEM_SIZE));
static double (*rhon_host)=(double*)malloc(sizeof(double)*(PROBLEM_SIZE));
static double (*rhos_host)=(double*)malloc(sizeof(double)*(PROBLEM_SIZE));
static double (*rhoq_host)=(double*)malloc(sizeof(double)*(PROBLEM_SIZE));
static double (*cuf_host)=(double*)malloc(sizeof(double)*(PROBLEM_SIZE));
static double (*q_host)=(double*)malloc(sizeof(double)*(PROBLEM_SIZE));
static double (*ue_host)[PROBLEM_SIZE]=(double(*)[PROBLEM_SIZE])malloc(sizeof(double)*((PROBLEM_SIZE)*(5)));
static double (*buf_host)[PROBLEM_SIZE]=(double(*)[PROBLEM_SIZE])malloc(sizeof(double)*((PROBLEM_SIZE)*(5)));
static double (*lhs_host)[IMAXP+1][5]=(double(*)[IMAXP+1][5])malloc(sizeof(double)*((IMAXP+1)*(IMAXP+1)*(5)));
static double (*lhsp_host)[IMAXP+1][5]=(double(*)[IMAXP+1][5])malloc(sizeof(double)*((IMAXP+1)*(IMAXP+1)*(5)));
static double (*lhsm_host)[IMAXP+1][5]=(double(*)[IMAXP+1][5])malloc(sizeof(double)*((IMAXP+1)*(IMAXP+1)*(5)));
#endif
static int grid_points[3];
static double dt_host;
double ce_host[13][5];
/* gpu variables */
static double* u_device;
static double* forcing_device;
static double* rhs_device;
static double* rho_i_device;
static double* us_device;
static double* vs_device;
static double* ws_device;
static double* qs_device;
static double* speed_device;
static double* square_device;
static double* lhs_device;
static double* rhs_buffer_device;
static double* rms_buffer_device;
static size_t size_u_device;
static size_t size_forcing_device;
static size_t size_rhs_device;
static size_t size_rho_i_device;
static size_t size_us_device;
static size_t size_vs_device;
static size_t size_ws_device;
static size_t size_qs_device;
static size_t size_speed_device;
static size_t size_square_device;
static size_t size_lhs_device;
static size_t size_rhs_buffer_device;
static size_t size_rms_buffer_device;
static int nx;
static int ny;
static int nz;
static int THREADS_PER_BLOCK_ON_ADD;
static int THREADS_PER_BLOCK_ON_COMPUTE_RHS_1;
static int THREADS_PER_BLOCK_ON_COMPUTE_RHS_2;
static int THREADS_PER_BLOCK_ON_ERROR_NORM_1;
static int THREADS_PER_BLOCK_ON_ERROR_NORM_2;
static int THREADS_PER_BLOCK_ON_EXACT_RHS_1;
static int THREADS_PER_BLOCK_ON_EXACT_RHS_2;
static int THREADS_PER_BLOCK_ON_EXACT_RHS_3;
static int THREADS_PER_BLOCK_ON_EXACT_RHS_4;
static int THREADS_PER_BLOCK_ON_INITIALIZE;
static int THREADS_PER_BLOCK_ON_RHS_NORM_1;
static int THREADS_PER_BLOCK_ON_RHS_NORM_2;
static int THREADS_PER_BLOCK_ON_TXINVR;
static int THREADS_PER_BLOCK_ON_X_SOLVE;
static int THREADS_PER_BLOCK_ON_Y_SOLVE;
static int THREADS_PER_BLOCK_ON_Z_SOLVE;
int gpu_device_id;
int total_devices;
hipDeviceProp_t gpu_device_properties;
extern __shared__ double extern_share_data[];

// device constants
__device__ __constant__ double tx1_device;
__device__ __constant__ double tx2_device;
__device__ __constant__ double tx3_device;
__device__ __constant__ double ty1_device;
__device__ __constant__ double ty2_device;
__device__ __constant__ double ty3_device;
__device__ __constant__ double tz1_device;
__device__ __constant__ double tz2_device;
__device__ __constant__ double tz3_device;
__device__ __constant__ double dx1_device;
__device__ __constant__ double dx2_device;
__device__ __constant__ double dx3_device;
__device__ __constant__ double dx4_device;
__device__ __constant__ double dx5_device;
__device__ __constant__ double dy1_device;
__device__ __constant__ double dy2_device;
__device__ __constant__ double dy3_device;
__device__ __constant__ double dy4_device;
__device__ __constant__ double dy5_device;
__device__ __constant__ double dz1_device;
__device__ __constant__ double dz2_device;
__device__ __constant__ double dz3_device;
__device__ __constant__ double dz4_device;
__device__ __constant__ double dz5_device;
__device__ __constant__ double dssp_device;
__device__ __constant__ double dxmax_device;
__device__ __constant__ double dymax_device;
__device__ __constant__ double dzmax_device;
__device__ __constant__ double xxcon1_device;
__device__ __constant__ double xxcon2_device;
__device__ __constant__ double xxcon3_device;
__device__ __constant__ double xxcon4_device;
__device__ __constant__ double xxcon5_device;
__device__ __constant__ double dx1tx1_device;
__device__ __constant__ double dx2tx1_device;
__device__ __constant__ double dx3tx1_device;
__device__ __constant__ double dx4tx1_device;
__device__ __constant__ double dx5tx1_device;
__device__ __constant__ double yycon1_device;
__device__ __constant__ double yycon2_device;
__device__ __constant__ double yycon3_device;
__device__ __constant__ double yycon4_device;
__device__ __constant__ double yycon5_device;
__device__ __constant__ double dy1ty1_device;
__device__ __constant__ double dy2ty1_device;
__device__ __constant__ double dy3ty1_device;
__device__ __constant__ double dy4ty1_device;
__device__ __constant__ double dy5ty1_device;
__device__ __constant__ double zzcon1_device;
__device__ __constant__ double zzcon2_device;
__device__ __constant__ double zzcon3_device;
__device__ __constant__ double zzcon4_device;
__device__ __constant__ double zzcon5_device;
__device__ __constant__ double dz1tz1_device;
__device__ __constant__ double dz2tz1_device;
__device__ __constant__ double dz3tz1_device;
__device__ __constant__ double dz4tz1_device;
__device__ __constant__ double dz5tz1_device;
__device__ __constant__ double dnxm1_device;
__device__ __constant__ double dnym1_device;
__device__ __constant__ double dnzm1_device;
__device__ __constant__ double c1c2_device;
__device__ __constant__ double c1c5_device;
__device__ __constant__ double c3c4_device;
__device__ __constant__ double c1345_device;
__device__ __constant__ double conz1_device;
__device__ __constant__ double c1_device;
__device__ __constant__ double c2_device;
__device__ __constant__ double c3_device;
__device__ __constant__ double c4_device;
__device__ __constant__ double c5_device;
__device__ __constant__ double c4dssp_device;
__device__ __constant__ double c5dssp_device;
__device__ __constant__ double dtdssp_device;
__device__ __constant__ double dttx1_device;
__device__ __constant__ double bt_device;
__device__ __constant__ double dttx2_device;
__device__ __constant__ double dtty1_device;
__device__ __constant__ double dtty2_device;
__device__ __constant__ double dttz1_device;
__device__ __constant__ double dttz2_device;
__device__ __constant__ double c2dttx1_device;
__device__ __constant__ double c2dtty1_device;
__device__ __constant__ double c2dttz1_device;
__device__ __constant__ double comz1_device;
__device__ __constant__ double comz4_device;
__device__ __constant__ double comz5_device;
__device__ __constant__ double comz6_device;
__device__ __constant__ double c3c4tx3_device;
__device__ __constant__ double c3c4ty3_device;
__device__ __constant__ double c3c4tz3_device;
__device__ __constant__ double c2iv_device;
__device__ __constant__ double con43_device;
__device__ __constant__ double con16_device;
__device__ __constant__ double dt_device;
__device__ __constant__ double ce_device[13][5];

/* function prototypes */
static void add_gpu();
__global__ static void add_gpu_kernel(double* u, 
		const double* rhs, 
		const int nx, 
		const int ny, 
		const int nz);
static void adi_gpu();
static void compute_rhs_gpu();
__global__ static void compute_rhs_gpu_kernel_1(double* rho_i, 
		double* us, 
		double* vs, 
		double* ws, 
		double* speed, 
		double* qs, 
		double* square, 
		const double* u, 
		const int nx, 
		const int ny, 
		const int nz);
__global__ static void compute_rhs_gpu_kernel_2(const double* rho_i, 
		const double* us, 
		const double* vs, 
		const double* ws, 
		const double* qs, 
		const double* square, 
		double* rhs, 
		const double* forcing, 
		const double* u, 
		const int nx, 
		const int ny, 
		const int nz);
static void error_norm_gpu(double rms[]);
__global__ static void error_norm_gpu_kernel_1(double* rms,
		const double* u,
		const int nx,
		const int ny,
		const int nz);
__global__ static void error_norm_gpu_kernel_2(double* rms,
		const int nx,
		const int ny,
		const int nz);	
static void exact_rhs_gpu();	
__global__ static void exact_rhs_gpu_kernel_1(double* forcing, 
		const int nx,
		const int ny,
		const int nz);
__global__ static void exact_rhs_gpu_kernel_2(double* forcing,
		const int nx,
		const int ny,
		const int nz);
__global__ static void exact_rhs_gpu_kernel_3(double* forcing,
		const int nx,
		const int ny,
		const int nz);
__global__ static void exact_rhs_gpu_kernel_4(double* forcing,
		const int nx,
		const int ny,
		const int nz);
__device__ static void exact_solution_gpu_device(const double xi,
		const double eta,
		const double zeta,
		double* dtemp);		
static void initialize_gpu();
__global__ static void initialize_gpu_kernel(double* u,
		const int nx,
		const int ny,
		const int nz);
static void release_gpu();
static void rhs_norm_gpu(double rms[]);
__global__ static void rhs_norm_gpu_kernel_1(double* rms,
		const double* rhs,
		const int nx,
		const int ny,
		const int nz);
__global__ static void rhs_norm_gpu_kernel_2(double* rms,
		const int nx,
		const int ny,
		const int nz);	
static void set_constants();
static void setup_gpu();
static void txinvr_gpu();
__global__ static void txinvr_gpu_kernel(const double* rho_i, 
		const double* us, 
		const double* vs, 
		const double* ws, 
		const double* speed, 
		const double* qs, 
		double* rhs, 
		const int nx, 
		const int ny, 
		const int nz);
static void verify_gpu(int no_time_steps,
		char* class_npb,
		boolean* verified);
static void x_solve_gpu();
__global__ static void x_solve_gpu_kernel(const double* rho_i, 
		const double* us, 
		const double* speed, 
		double* rhs, 
		double* lhs, 
		double* rhstmp, 
		const int nx, 
		const int ny, 
		const int nz);
static void y_solve_gpu();
__global__ static void y_solve_gpu_kernel(const double* rho_i, 
		const double* vs, 
		const double* speed, 
		double* rhs, 
		double* lhs, 
		double* rhstmp, 
		const int nx, 
		const int ny, 
		const int nz);
static void z_solve_gpu();
__global__ static void z_solve_gpu_kernel(const double* rho_i,
		const double* us,
		const double* vs,
		const double* ws,
		const double* speed,
		const double* qs,
		const double* u,
		double* rhs,
		double* lhs,
		double* rhstmp,
		const int nx,
		const int ny,
		const int nz);

/* sp */
int main(int argc, char** argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
#if defined(PROFILING)
	printf(" PROFILING mode on\n");
#endif
	int niter, step, n3;
	double mflops, t, tmax;
	boolean verified;
	char class_npb;
	/*
	 * ---------------------------------------------------------------------
	 * sp parameters
	 * ---------------------------------------------------------------------
	 */
	niter=NITER_DEFAULT;
	dt_host=DT_DEFAULT;
	grid_points[0]=PROBLEM_SIZE;
	grid_points[1]=PROBLEM_SIZE;
	grid_points[2]=PROBLEM_SIZE;
	printf("\n\n NAS Parallel Benchmarks 4.1 HIP C++ version - SP Benchmark\n\n");
	printf(" Size: %4dx%4dx%4d\n",grid_points[0],grid_points[1],grid_points[2]);
	printf(" Iterations: %4d    dt: %10.6f\n",niter,dt_host);
	printf("\n");
	if((grid_points[0]>IMAX)||(grid_points[1]>JMAX)||(grid_points[2]>KMAX)){
		printf(" %d, %d, %d\n",grid_points[0],grid_points[1],grid_points[2]);
		printf(" Problem size too big for compiled array sizes\n");
		return 0;
	}
	nx=grid_points[0];
	ny=grid_points[1];
	nz=grid_points[2];
	setup_gpu();
	set_constants();
	timer_clear(PROFILING_TOTAL_TIME);
#if defined(PROFILING)
	timer_clear(PROFILING_ADD);
	timer_clear(PROFILING_COMPUTE_RHS_1);
	timer_clear(PROFILING_COMPUTE_RHS_2);
	timer_clear(PROFILING_ERROR_NORM_1);
	timer_clear(PROFILING_ERROR_NORM_2);
	timer_clear(PROFILING_EXACT_RHS_1);
	timer_clear(PROFILING_EXACT_RHS_2);
	timer_clear(PROFILING_EXACT_RHS_3);
	timer_clear(PROFILING_EXACT_RHS_4);
	timer_clear(PROFILING_INITIALIZE);
	timer_clear(PROFILING_RHS_NORM_1);
	timer_clear(PROFILING_RHS_NORM_2);
	timer_clear(PROFILING_TXINVR);
	timer_clear(PROFILING_X_SOLVE);
	timer_clear(PROFILING_Y_SOLVE);
	timer_clear(PROFILING_Z_SOLVE);
#endif
	exact_rhs_gpu();
	initialize_gpu();
	/*
	 * ---------------------------------------------------------------------
	 * do one time step to touch all code, and reinitialize
	 * ---------------------------------------------------------------------
	 */
	adi_gpu();
	initialize_gpu();
	timer_clear(PROFILING_TOTAL_TIME);
#if defined(PROFILING)
	timer_clear(PROFILING_ADD);
	timer_clear(PROFILING_COMPUTE_RHS_1);
	timer_clear(PROFILING_COMPUTE_RHS_2);
	timer_clear(PROFILING_ERROR_NORM_1);
	timer_clear(PROFILING_ERROR_NORM_2);
	timer_clear(PROFILING_EXACT_RHS_1);
	timer_clear(PROFILING_EXACT_RHS_2);
	timer_clear(PROFILING_EXACT_RHS_3);
	timer_clear(PROFILING_EXACT_RHS_4);
	timer_clear(PROFILING_INITIALIZE);
	timer_clear(PROFILING_RHS_NORM_1);
	timer_clear(PROFILING_RHS_NORM_2);
	timer_clear(PROFILING_TXINVR);
	timer_clear(PROFILING_X_SOLVE);
	timer_clear(PROFILING_Y_SOLVE);
	timer_clear(PROFILING_Z_SOLVE);
#endif
	timer_start(PROFILING_TOTAL_TIME);/*#start_timer*/
	for(step=1;step<=niter;step++){
		if((step%20)==0||step==1){printf(" Time step %4d\n",step);}
		adi_gpu();
	}
	timer_stop(PROFILING_TOTAL_TIME);/*#stop_timer*/
	tmax=timer_read(PROFILING_TOTAL_TIME);
	verify_gpu(niter, &class_npb, &verified);
	if(tmax!=0.0){
		n3=grid_points[0]*grid_points[1]*grid_points[2];
		t=(grid_points[0]+grid_points[1]+grid_points[2])/3.0;
		mflops=(881.174*(double)n3-
				4683.91*(t*t)+
				11484.5*t-
				19272.4)*(double)niter/(tmax*1000000.0);
	}else{
		mflops=0.0;
	}
	char gpu_config[256];
	char gpu_config_string[2048];
#if defined(PROFILING)
	sprintf(gpu_config, "%5s\t%25s\t%25s\t%25s\n", "GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-add", THREADS_PER_BLOCK_ON_ADD, timer_read(PROFILING_ADD), (timer_read(PROFILING_ADD)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-compute-rhs-1", THREADS_PER_BLOCK_ON_COMPUTE_RHS_1, timer_read(PROFILING_COMPUTE_RHS_1), (timer_read(PROFILING_COMPUTE_RHS_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-compute-rhs-2", THREADS_PER_BLOCK_ON_COMPUTE_RHS_2, timer_read(PROFILING_COMPUTE_RHS_2), (timer_read(PROFILING_COMPUTE_RHS_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-error-norm-1", THREADS_PER_BLOCK_ON_ERROR_NORM_1, timer_read(PROFILING_ERROR_NORM_1), (timer_read(PROFILING_ERROR_NORM_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-error-norm-2", THREADS_PER_BLOCK_ON_ERROR_NORM_2, timer_read(PROFILING_ERROR_NORM_2), (timer_read(PROFILING_ERROR_NORM_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-exact-rhs-1", THREADS_PER_BLOCK_ON_EXACT_RHS_1, timer_read(PROFILING_EXACT_RHS_1), (timer_read(PROFILING_EXACT_RHS_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-exact-rhs-2", THREADS_PER_BLOCK_ON_EXACT_RHS_2, timer_read(PROFILING_EXACT_RHS_2), (timer_read(PROFILING_EXACT_RHS_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-exact-rhs-3", THREADS_PER_BLOCK_ON_EXACT_RHS_3, timer_read(PROFILING_EXACT_RHS_3), (timer_read(PROFILING_EXACT_RHS_3)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-exact-rhs-4", THREADS_PER_BLOCK_ON_EXACT_RHS_4, timer_read(PROFILING_EXACT_RHS_4), (timer_read(PROFILING_EXACT_RHS_4)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-initialize", THREADS_PER_BLOCK_ON_INITIALIZE, timer_read(PROFILING_INITIALIZE), (timer_read(PROFILING_INITIALIZE)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-rhs-norm-1", THREADS_PER_BLOCK_ON_RHS_NORM_1, timer_read(PROFILING_RHS_NORM_1), (timer_read(PROFILING_RHS_NORM_1)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-rhs-norm-2", THREADS_PER_BLOCK_ON_RHS_NORM_2, timer_read(PROFILING_RHS_NORM_2), (timer_read(PROFILING_RHS_NORM_2)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-txinvr", THREADS_PER_BLOCK_ON_TXINVR, timer_read(PROFILING_TXINVR), (timer_read(PROFILING_TXINVR)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-x-solve", THREADS_PER_BLOCK_ON_X_SOLVE, timer_read(PROFILING_X_SOLVE), (timer_read(PROFILING_X_SOLVE)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-y-solve", THREADS_PER_BLOCK_ON_Y_SOLVE, timer_read(PROFILING_Y_SOLVE), (timer_read(PROFILING_Y_SOLVE)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " sp-z-solve", THREADS_PER_BLOCK_ON_Z_SOLVE, timer_read(PROFILING_Z_SOLVE), (timer_read(PROFILING_Z_SOLVE)*100/timer_read(PROFILING_TOTAL_TIME)));
	strcat(gpu_config_string, gpu_config);
#else
	sprintf(gpu_config, "%5s\t%25s\n", "GPU Kernel", "Threads Per Block");
	strcpy(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-add", THREADS_PER_BLOCK_ON_ADD);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-compute-rhs-1", THREADS_PER_BLOCK_ON_COMPUTE_RHS_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-compute-rhs-2", THREADS_PER_BLOCK_ON_COMPUTE_RHS_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-error-norm-1", THREADS_PER_BLOCK_ON_ERROR_NORM_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-error-norm-2", THREADS_PER_BLOCK_ON_ERROR_NORM_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-exact-rhs-1", THREADS_PER_BLOCK_ON_EXACT_RHS_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-exact-rhs-2", THREADS_PER_BLOCK_ON_EXACT_RHS_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-exact-rhs-3", THREADS_PER_BLOCK_ON_EXACT_RHS_3);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-exact-rhs-4", THREADS_PER_BLOCK_ON_EXACT_RHS_4);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-initialize", THREADS_PER_BLOCK_ON_INITIALIZE);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-rhs-norm-1", THREADS_PER_BLOCK_ON_RHS_NORM_1);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-rhs-norm-2", THREADS_PER_BLOCK_ON_RHS_NORM_2);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-txinvr", THREADS_PER_BLOCK_ON_TXINVR);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-x-solve", THREADS_PER_BLOCK_ON_X_SOLVE);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-y-solve", THREADS_PER_BLOCK_ON_Y_SOLVE);
	strcat(gpu_config_string, gpu_config);
	sprintf(gpu_config, "%29s\t%25d\n", " sp-z-solve", THREADS_PER_BLOCK_ON_Z_SOLVE);
	strcat(gpu_config_string, gpu_config);
#endif
	c_print_results((char*)"SP",
			class_npb,
			grid_points[0],
			grid_points[1],
			grid_points[2],
			niter,
			tmax,
			mflops,
			(char*)"          floating point",
			verified,
			(char*)NPBVERSION,
			(char*)COMPILETIME,
			(char*)COMPILERVERSION,
			(char*)LIBVERSION,
			(char*)CPU_MODEL,
			(char*)gpu_device_properties.name,
			gpu_config_string,
			(char*)CS1,
			(char*)CS2,
			(char*)CS3,
			(char*)CS4,
			(char*)CS5,
			(char*)CS6,
			(char*)"(none)");
	release_gpu();
	return 0;
}

/*
 * ---------------------------------------------------------------------
 * addition of update to the vector u
 * ---------------------------------------------------------------------
 */
static void add_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_ADD);
#endif
	/* #KERNEL ADD */
	int add_workload = nx * ny * nz;
	int add_threads_per_block = THREADS_PER_BLOCK_ON_ADD;
	int add_blocks_per_grid = (ceil((double)add_workload/(double)add_threads_per_block));

	add_gpu_kernel<<<
		add_blocks_per_grid, 
		add_threads_per_block>>>(
				u_device, 
				rhs_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_ADD);
#endif
}

/*
 * ---------------------------------------------------------------------
 * addition of update to the vector u
 * ---------------------------------------------------------------------
 */
__global__ static void add_gpu_kernel(double* u,
		const double* rhs,
		const int nx,
		const int ny,
		const int nz){
	int i_j_k, i, j, k;

	i_j_k = blockIdx.x * blockDim.x + threadIdx.x;

	i = i_j_k % nx;
	j = (i_j_k / nx) % ny;
	k = i_j_k / (nx * ny);

	if(i_j_k >= (nx*ny*nz)){
		return;
	}

	/* array(m,i,j,k) */
	u(0,i,j,k)+=rhs(0,i,j,k);
	u(1,i,j,k)+=rhs(1,i,j,k);
	u(2,i,j,k)+=rhs(2,i,j,k);
	u(3,i,j,k)+=rhs(3,i,j,k);
	u(4,i,j,k)+=rhs(4,i,j,k);
}

static void adi_gpu(){
	compute_rhs_gpu();
	txinvr_gpu();
	x_solve_gpu();
	y_solve_gpu();
	z_solve_gpu();
	add_gpu();
}

static void compute_rhs_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_COMPUTE_RHS_1);
#endif
	/* #KERNEL COMPUTE RHS 1 */
	int compute_rhs_1_workload = nx * ny * nz;
	int compute_rhs_1_threads_per_block = THREADS_PER_BLOCK_ON_COMPUTE_RHS_1;
	int compute_rhs_1_blocks_per_grid = (ceil((double)compute_rhs_1_workload/(double)compute_rhs_1_threads_per_block));

	compute_rhs_gpu_kernel_1<<<
		compute_rhs_1_blocks_per_grid,
		compute_rhs_1_threads_per_block>>>(
				rho_i_device, 
				us_device, 
				vs_device, 
				ws_device, 
				speed_device, 
				qs_device, 
				square_device, 
				u_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_COMPUTE_RHS_1);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_COMPUTE_RHS_2);
#endif
	/* #KERNEL COMPUTE RHS 2 */
	int compute_rhs_2_threads_per_block;
	dim3 compute_rhs_2_blocks_per_grid(ny, nz);
	if(THREADS_PER_BLOCK_ON_COMPUTE_RHS_2 != nx){
		compute_rhs_2_threads_per_block = nx;
	}
	else{
		compute_rhs_2_threads_per_block = THREADS_PER_BLOCK_ON_COMPUTE_RHS_2;
	}

	compute_rhs_gpu_kernel_2<<<
		compute_rhs_2_blocks_per_grid, 
		compute_rhs_2_threads_per_block>>>(
				rho_i_device, 
				us_device, 
				vs_device, 
				ws_device, 
				qs_device, 
				square_device, 
				rhs_device, 
				forcing_device, 
				u_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_COMPUTE_RHS_2);
#endif	
}

__global__ static void compute_rhs_gpu_kernel_1(double* rho_i,
		double* us,
		double* vs, 
		double* ws, 
		double* speed, 
		double* qs, 
		double* square, 
		const double* u, 
		const int nx, 
		const int ny, 
		const int nz){
	int i_j_k, i, j, k;

	i_j_k = blockIdx.x * blockDim.x + threadIdx.x;

	i = i_j_k % nx;
	j = (i_j_k / nx) % ny;
	k = i_j_k / (nx * ny);

	if(i_j_k >= (nx*ny*nz)){
		return;
	}

	/*
	 * ---------------------------------------------------------------------
	 * compute the reciprocal of density, and the kinetic energy, 
	 * and the speed of sound. 
	 * ---------------------------------------------------------------------
	 */
	double rho_inv=1.0/u(0,i,j,k);
	double square_ijk;
	rho_i(i,j,k)=rho_inv;
	us(i,j,k)=u(1,i,j,k)*rho_inv;
	vs(i,j,k)=u(2,i,j,k)*rho_inv;
	ws(i,j,k)=u(3,i,j,k)*rho_inv;
	square(i,j,k)=square_ijk=0.5*(u(1,i,j,k)*u(1,i,j,k)+u(2,i,j,k)*u(2,i,j,k)+u(3,i,j,k)*u(3,i,j,k))*rho_inv;
	qs(i,j,k)=square_ijk*rho_inv;
	/*
	 * ---------------------------------------------------------------------
	 * (don't need speed and ainx until the lhs computation)
	 * ---------------------------------------------------------------------
	 */
	speed(i,j,k)=sqrt(c1c2_device*rho_inv*(u(4,i,j,k)-square_ijk));
}

__global__ static void compute_rhs_gpu_kernel_2(const double* rho_i, 
		const double* us, 
		const double* vs, 
		const double* ws, 
		const double* qs, 
		const double* square, 
		double* rhs, 
		const double* forcing, 
		const double* u, 
		const int nx, 
		const int ny, 
		const int nz){
	int i, j, k, m;

	k=blockIdx.y;
	j=blockIdx.x;
	i=threadIdx.x;

	double rtmp[5];
	/*
	 * ---------------------------------------------------------------------
	 * copy the exact forcing term to the right hand side;  because 
	 * this forcing term is known, we can store it on the whole grid
	 * including the boundary                   
	 * ---------------------------------------------------------------------
	 */
	for(m=0;m<5;m++){rtmp[m]=forcing(m,i,j,k);}
	/*
	 * ---------------------------------------------------------------------
	 * compute xi-direction fluxes 
	 * ---------------------------------------------------------------------
	 */
	if(k>=1 && k<nz-1 && j>=1 && j<ny-1 && i>=1 && i<nx-1){
		double uijk=us(i,j,k);
		double up1=us(i+1,j,k);
		double um1=us(i-1,j,k);
		rtmp[0]=rtmp[0]+dx1tx1_device*(u(0,i+1,j,k)-2.0*u(0,i,j,k)+u(0,i-1,j,k))-tx2_device*(u(1,i+1,j,k)-u(1,i-1,j,k));
		rtmp[1]=rtmp[1]+dx2tx1_device*(u(1,i+1,j,k)-2.0*u(1,i,j,k)+u(1,i-1,j,k))+xxcon2_device*con43_device*(up1-2.0*uijk+um1)-tx2_device*(u(1,i+1,j,k)*up1-u(1,i-1,j,k)*um1+(u(4,i+1,j,k)-square(i+1,j,k)-u(4,i-1,j,k)+square(i-1,j,k))*c2_device);
		rtmp[2]=rtmp[2]+dx3tx1_device*(u(2,i+1,j,k)-2.0*u(2,i,j,k)+u(2,i-1,j,k))+xxcon2_device*(vs(i+1,j,k)-2.0*vs(i,j,k)+vs(i-1,j,k))-tx2_device*(u(2,i+1,j,k)*up1-u(2,i-1,j,k)*um1);
		rtmp[3]=rtmp[3]+dx4tx1_device*(u(3,i+1,j,k)-2.0*u(3,i,j,k)+u(3,i-1,j,k))+xxcon2_device*(ws(i+1,j,k)-2.0*ws(i,j,k)+ws(i-1,j,k))-tx2_device*(u(3,i+1,j,k)*up1-u(3,i-1,j,k)*um1);
		rtmp[4]=rtmp[4]+dx5tx1_device*(u(4,i+1,j,k)-2.0*u(4,i,j,k)+u(4,i-1,j,k))+xxcon3_device*(qs(i+1,j,k)-2.0*qs(i,j,k)+qs(i-1,j,k))+ xxcon4_device*(up1*up1-2.0*uijk*uijk+um1*um1)+xxcon5_device*(u(4,i+1,j,k)*rho_i(i+1,j,k)-2.0*u(4,i,j,k)*rho_i(i,j,k)+u(4,i-1,j,k)*rho_i(i-1,j,k))-tx2_device*((c1_device*u(4,i+1,j,k)-c2_device*square(i+1,j,k))*up1-(c1_device*u(4,i-1,j,k)-c2_device*square(i-1,j,k))*um1);
		/*
		 * ---------------------------------------------------------------------
		 * add fourth order xi-direction dissipation               
		 * ---------------------------------------------------------------------
		 */
		if(i==1){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(5.0*u(m,i,j,k)-4.0*u(m,i+1,j,k)+u(m,i+2,j,k));}
		}else if(i==2){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(-4.0*u(m,i-1,j,k)+6.0*u(m,i,j,k)-4.0*u(m,i+1,j,k)+u(m,i+2,j,k));}
		}else if(i>=3 && i<nx-3){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(u(m,i-2,j,k)-4.0*u(m,i-1,j,k)+6.0*u(m,i,j,k)-4.0*u(m,i+1,j,k)+u(m,i+2,j,k));}
		}else if(i==nx-3){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(u(m,i-2,j,k)-4.0*u(m,i-1,j,k)+6.0*u(m,i,j,k)-4.0*u(m,i+1,j,k));}
		}else if(i==nx-2){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(u(m,i-2,j,k)-4.0*u(m,i-1,j,k) + 5.0*u(m,i,j,k));}
		}
		/*
		 * ---------------------------------------------------------------------
		 * compute eta-direction fluxes 
		 * ---------------------------------------------------------------------
		 */
		double vijk=vs(i,j,k);
		double vp1=vs(i,j+1,k);
		double vm1=vs(i,j-1,k);
		rtmp[0]=rtmp[0]+dy1ty1_device*(u(0,i,j+1,k)-2.0*u(0,i,j,k)+u(0,i,j-1,k))-ty2_device*(u(2,i,j+1,k)-u(2,i,j-1,k));
		rtmp[1]=rtmp[1]+dy2ty1_device*(u(1,i,j+1,k)-2.0*u(1,i,j,k)+u(1,i,j-1,k))+yycon2_device*(us(i,j+1,k)-2.0*us(i,j,k)+us(i,j-1,k))-ty2_device*(u(1,i,j+1,k)*vp1-u(1,i,j-1,k)*vm1);
		rtmp[2]=rtmp[2]+dy3ty1_device*(u(2,i,j+1,k)-2.0*u(2,i,j,k)+u(2,i,j-1,k))+yycon2_device*con43_device*(vp1-2.0*vijk+vm1)-ty2_device*(u(2,i,j+1,k)*vp1-u(2,i,j-1,k)*vm1+(u(4,i,j+1,k)-square(i,j+1,k)-u(4,i,j-1,k)+square(i,j-1,k))*c2_device);
		rtmp[3]=rtmp[3]+dy4ty1_device*(u(3,i,j+1,k)-2.0*u(3,i,j,k)+u(3,i,j-1,k))+yycon2_device*(ws(i,j+1,k)-2.0*ws(i,j,k)+ws(i,j-1,k))-ty2_device*(u(3,i,j+1,k)*vp1-u(3,i,j-1,k)*vm1);
		rtmp[4]=rtmp[4]+dy5ty1_device*(u(4,i,j+1,k)-2.0*u(4,i,j,k)+u(4,i,j-1,k))+yycon3_device*(qs(i,j+1,k)-2.0*qs(i,j,k)+qs(i,j-1,k))+yycon4_device*(vp1*vp1-2.0*vijk*vijk+vm1*vm1)+yycon5_device*(u(4,i,j+1,k)*rho_i(i,j+1,k)-2.0*u(4,i,j,k)*rho_i(i,j,k)+u(4,i,j-1,k)*rho_i(i,j-1,k))-ty2_device*((c1_device*u(4,i,j+1,k)-c2_device*square(i,j+1,k))*vp1-(c1_device*u(4,i,j-1,k)-c2_device*square(i,j-1,k))*vm1);
		/*
		 * ---------------------------------------------------------------------
		 * add fourth order eta-direction dissipation         
		 * ---------------------------------------------------------------------
		 */
		if(j==1){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(5.0*u(m,i,j,k)-4.0*u(m,i,j+1,k)+u(m,i,j+2,k));}
		}else if(j==2){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(-4.0*u(m,i,j-1,k)+6.0*u(m,i,j,k)-4.0*u(m,i,j+1,k)+u(m,i,j+2,k));}
		}else if(j>=3 && j<ny-3){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(u(m,i,j-2,k)-4.0*u(m,i,j-1,k)+6.0*u(m,i,j,k)-4.0*u(m,i,j+1,k)+u(m,i,j+2,k));}
		}else if(j==ny-3){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(u(m,i,j-2,k)-4.0*u(m,i,j-1,k)+6.0*u(m,i,j,k)-4.0*u(m,i,j+1,k));}
		}else if(j==ny-2){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(u(m,i,j-2,k)-4.0*u(m,i,j-1,k)+5.0*u(m,i,j,k));}
		}
		/*
		 * ---------------------------------------------------------------------
		 * compute zeta-direction fluxes 
		 * ---------------------------------------------------------------------
		 */
		double wijk=ws(i,j,k);
		double wp1=ws(i,j,k+1);
		double wm1=ws(i,j,k-1);
		rtmp[0]=rtmp[0]+dz1tz1_device*(u(0,i,j,k+1)-2.0*u(0,i,j,k)+u(0,i,j,k-1))-tz2_device*(u(3,i,j,k+1)-u(3,i,j,k-1));
		rtmp[1]=rtmp[1]+dz2tz1_device*(u(1,i,j,k+1)-2.0*u(1,i,j,k)+u(1,i,j,k-1))+zzcon2_device*(us(i,j,k+1)-2.0*us(i,j,k)+us(i,j,k-1))-tz2_device*(u(1,i,j,k+1)*wp1-u(1,i,j,k-1)*wm1);
		rtmp[2]=rtmp[2]+dz3tz1_device*(u(2,i,j,k+1)-2.0*u(2,i,j,k)+u(2,i,j,k-1))+zzcon2_device*(vs(i,j,k+1)-2.0*vs(i,j,k)+vs(i,j,k-1))-tz2_device*(u(2,i,j,k+1)*wp1-u(2,i,j,k-1)*wm1);
		rtmp[3]=rtmp[3]+dz4tz1_device*(u(3,i,j,k+1)-2.0*u(3,i,j,k)+u(3,i,j,k-1))+zzcon2_device*con43_device*(wp1-2.0*wijk+wm1)-tz2_device*(u(3,i,j,k+1)*wp1-u(3,i,j,k-1)*wm1+(u(4,i,j,k+1)-square(i,j,k+1)-u(4,i,j,k-1)+square(i,j,k-1))*c2_device);
		rtmp[4]=rtmp[4]+dz5tz1_device*(u(4,i,j,k+1)-2.0*u(4,i,j,k)+u(4,i,j,k-1))+zzcon3_device*(qs(i,j,k+1)-2.0*qs(i,j,k)+qs(i,j,k-1))+zzcon4_device*(wp1*wp1-2.0*wijk*wijk+wm1*wm1)+zzcon5_device*(u(4,i,j,k+1)*rho_i(i,j,k+1)-2.0*u(4,i,j,k)*rho_i(i,j,k)+u(4,i,j,k-1)*rho_i(i,j,k-1))-tz2_device*((c1_device*u(4,i,j,k+1)-c2_device*square(i,j,k+1))*wp1-(c1_device*u(4,i,j,k-1)-c2_device*square(i,j,k-1))*wm1);
		/*
		 * ---------------------------------------------------------------------
		 * add fourth order zeta-direction dissipation                
		 * ---------------------------------------------------------------------
		 */
		if(k==1){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(5.0*u(m,i,j,k)-4.0*u(m,i,j,k+1)+u(m,i,j,k+2));}
		}else if(k==2){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(-4.0*u(m,i,j,k-1)+6.0*u(m,i,j,k)-4.0*u(m,i,j,k+1)+u(m,i,j,k+2));}
		}else if(k>=3 && k<nz-3){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(u(m,i,j,k-2)-4.0*u(m,i,j,k-1)+6.0*u(m,i,j,k)-4.0*u(m,i,j,k+1)+u(m,i,j,k+2));}
		}else if(k==nz-3){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(u(m,i,j,k-2)-4.0*u(m,i,j,k-1)+6.0*u(m,i,j,k)-4.0*u(m,i,j,k+1));}
		}else if(k==nz-2){
			for(m=0;m<5;m++){rtmp[m]=rtmp[m]-dssp_device*(u(m,i,j,k-2)-4.0*u(m,i,j,k-1)+5.0*u(m,i,j,k));}
		}
		for(m=0;m<5;m++){rtmp[m]*=dt_device;}
	}
	for(m=0;m<5;m++){rhs(m,i,j,k)=rtmp[m];}
}

/*
 * ---------------------------------------------------------------------
 * this function computes the norm of the difference between the
 * computed solution and the exact solution
 * ---------------------------------------------------------------------
 */
static void error_norm_gpu(double rms[]){
#if defined(PROFILING)
	timer_start(PROFILING_ERROR_NORM_1);
#endif
	/* #KERNEL ERROR NORM 1 */
	int error_norm_1_threads_per_block = THREADS_PER_BLOCK_ON_ERROR_NORM_1;
	dim3 error_norm_1_blocks_per_grid(ny, nx);

	error_norm_gpu_kernel_1<<<
		error_norm_1_blocks_per_grid, 
		error_norm_1_threads_per_block>>>(
				rms_buffer_device, 
				u_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_ERROR_NORM_1);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_ERROR_NORM_2);
#endif
	/* #KERNEL ERROR NORM 2 */
	int error_norm_2_threads_per_block = THREADS_PER_BLOCK_ON_ERROR_NORM_2;
	int error_norm_2_blocks_per_grid = 1;

	error_norm_gpu_kernel_2<<<
		error_norm_2_blocks_per_grid,
		error_norm_2_threads_per_block,
		sizeof(double)*error_norm_2_threads_per_block*5>>>(
				rms_buffer_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_ERROR_NORM_2);
#endif

	hipMemcpy(rms, rms_buffer_device, 5*sizeof(double), hipMemcpyDeviceToHost);
}

__global__ static void error_norm_gpu_kernel_1(double* rms,
		const double* u,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double xi, eta, zeta, u_exact[5], rms_loc[5];

	j=blockIdx.x*blockDim.x+threadIdx.x;
	i=blockIdx.y*blockDim.y+threadIdx.y;

	if(j>=ny || i>=nx){return;}

	for(m=0;m<5;m++){rms_loc[m]=0.0;}
	xi=(double)i*dnxm1_device;
	eta=(double)j*dnym1_device;
	for(k=0; k<nz; k++){
		zeta=(double)k*dnzm1_device;
		exact_solution_gpu_device(xi, eta, zeta, u_exact);
		for(m=0; m<5; m++){
			double add=u(m,i,j,k)-u_exact[m];
			rms_loc[m]+=add*add;
		}
	}
	for(m=0;m<5;m++){rms[i+nx*(j+ny*m)]=rms_loc[m];}
}

__global__ static void error_norm_gpu_kernel_2(double* rms,
		const int nx,
		const int ny,
		const int nz){
	int i, m, maxpos, dist;

	double* buffer = (double*)extern_share_data;

	i = threadIdx.x;

	for(m=0;m<5;m++){buffer[i+(m*blockDim.x)]=0.0;}
	while(i<nx*ny){
		for(m=0;m<5;m++){buffer[threadIdx.x+(m*blockDim.x)]+=rms[i+nx*ny*m];}
		i+=blockDim.x;
	}
	maxpos=blockDim.x;
	dist=(maxpos+1)/2;
	i=threadIdx.x;
	__syncthreads();
	while(maxpos>1){
		if(i<dist && i+dist<maxpos){
			for(m=0;m<5;m++){buffer[i+(m*blockDim.x)]+=buffer[(i+dist)+(m*blockDim.x)];}
		}
		maxpos=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	m=threadIdx.x;
	if(m<5){rms[m]=sqrt(buffer[0+(m*blockDim.x)]/((double)(nz-2)*(double)(ny-2)*(double)(nx-2)));}
}

/*
 * ---------------------------------------------------------------------
 * compute the right hand side based on exact solution
 * ---------------------------------------------------------------------
 */
static void exact_rhs_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_EXACT_RHS_1);
#endif
	/* #KERNEL EXACT RHS 1 */
	int rhs1_workload = nx * ny * nz;
	int rhs1_threads_per_block = THREADS_PER_BLOCK_ON_EXACT_RHS_1;
	int rhs1_blocks_per_grid = (ceil((double)rhs1_workload/(double)rhs1_threads_per_block));

	exact_rhs_gpu_kernel_1<<<
		rhs1_blocks_per_grid,
		rhs1_threads_per_block>>>(
				forcing_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_EXACT_RHS_1);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_EXACT_RHS_2);
#endif
	/* #KERNEL EXACT RHS 2 */
	int rhs2_threads_per_block;
	dim3 rhs2_blocks_per_grid(nz, ny);
	if(THREADS_PER_BLOCK_ON_EXACT_RHS_2 > nx){
		rhs2_threads_per_block = nx;
	}
	else{
		rhs2_threads_per_block = THREADS_PER_BLOCK_ON_EXACT_RHS_2;
	}

	exact_rhs_gpu_kernel_2<<<
		rhs2_blocks_per_grid,
		rhs2_threads_per_block>>>(
				forcing_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_EXACT_RHS_2);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_EXACT_RHS_3);
#endif			
	/* #KERNEL EXACT RHS 3 */
	int rhs3_threads_per_block;
	dim3 rhs3_blocks_per_grid(nz, nx);
	if(THREADS_PER_BLOCK_ON_EXACT_RHS_3 > ny){
		rhs3_threads_per_block = ny;
	}
	else{
		rhs3_threads_per_block = THREADS_PER_BLOCK_ON_EXACT_RHS_3;
	}

	exact_rhs_gpu_kernel_3<<<
		rhs3_blocks_per_grid, 
		rhs3_threads_per_block>>>(
				forcing_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_EXACT_RHS_3);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_EXACT_RHS_4);
#endif
	/* #KERNEL EXACT RHS 4 */
	int rhs4_threads_per_block;
	dim3 rhs4_blocks_per_grid(ny, nx);
	if(THREADS_PER_BLOCK_ON_EXACT_RHS_4 > nz){
		rhs4_threads_per_block = nz;
	}
	else{
		rhs4_threads_per_block = THREADS_PER_BLOCK_ON_EXACT_RHS_4;
	}

	exact_rhs_gpu_kernel_4<<<
		rhs4_blocks_per_grid, 
		rhs4_threads_per_block>>>(
				forcing_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_EXACT_RHS_4);
#endif
}

__global__ static void exact_rhs_gpu_kernel_1(double* forcing, 
		const int nx,
		const int ny,
		const int nz){
	int i_j_k, i, j, k;

	i_j_k = blockIdx.x * blockDim.x + threadIdx.x;

	i = i_j_k % nx;
	j = (i_j_k / nx) % ny;
	k = i_j_k / (nx * ny);

	if(i_j_k >= (nx*ny*nz)){
		return;
	}

	/*
	 * ---------------------------------------------------------------------
	 * initialize                                  
	 * ---------------------------------------------------------------------
	 */
	/* array(m,i,j,k) */
	forcing(0,i,j,k)=0.0;
	forcing(1,i,j,k)=0.0;
	forcing(2,i,j,k)=0.0;
	forcing(3,i,j,k)=0.0;
	forcing(4,i,j,k)=0.0;
}

__global__ static void exact_rhs_gpu_kernel_2(double* forcing,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double xi, eta, zeta, dtemp[5], dtpp;
	double ue[5][5], buf[3][5], cuf[3], q[3];

	k=blockIdx.x*blockDim.x+threadIdx.x+1;
	j=blockIdx.y*blockDim.y+threadIdx.y+1;

	if(k>=(nz-1) || j>=(ny-1)){return;}

	zeta=(double)k*dnzm1_device;
	eta=(double)j*dnym1_device;
	/*
	 * ---------------------------------------------------------------------
	 * xi-direction flux differences                      
	 * ---------------------------------------------------------------------
	 */
	for(i=0; i<3; i++){
		xi=(double)i*dnxm1_device;
		exact_solution_gpu_device(xi, eta, zeta, dtemp);
		for(m=0;m<5;m++){ue[i+1][m]=dtemp[m];}
		dtpp=1.0/dtemp[0];
		for(m=1;m<5;m++){buf[i][m]=dtpp*dtemp[m];}
		cuf[i]=buf[i][1]*buf[i][1];
		buf[i][0]=cuf[i]+buf[i][2]*buf[i][2]+buf[i][3]*buf[i][3];
		q[i]=0.5*(buf[i][1]*ue[i+1][1]+buf[i][2]*ue[i+1][2]+buf[i][3]*ue[i+1][3]);
	}
	for(i=1; i<nx-1; i++){
		if(i+2<nx){
			xi=(double)(i+2)*dnxm1_device;
			exact_solution_gpu_device(xi, eta, zeta, dtemp);
			for(m=0;m<5;m++){ue[4][m]=dtemp[m];}
		}
		dtemp[0]=0.0-tx2_device*(ue[3][1]-ue[1][1])+dx1tx1_device*(ue[3][0]-2.0*ue[2][0]+ue[1][0]);
		dtemp[1]=0.0-tx2_device*((ue[3][1]*buf[2][1]+c2_device*(ue[3][4]-q[2]))-(ue[1][1]*buf[0][1]+c2_device*(ue[1][4]-q[0])))+xxcon1_device*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dx2tx1_device*(ue[3][1]-2.0*ue[2][1]+ue[1][1]);
		dtemp[2]=0.0-tx2_device*(ue[3][2]*buf[2][1]-ue[1][2]*buf[0][1])+xxcon2_device*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dx3tx1_device*(ue[3][2]-2.0*ue[2][2]+ue[1][2]);
		dtemp[3]=0.0-tx2_device*(ue[3][3]*buf[2][1]-ue[1][3]*buf[0][1])+xxcon2_device*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dx4tx1_device*(ue[3][3]-2.0*ue[2][3]+ue[1][3]);
		dtemp[4]=0.0-tx2_device*(buf[2][1]*(c1_device*ue[3][4]-c2_device*q[2])-buf[0][1]*(c1_device*ue[1][4]-c2_device*q[0]))+0.5*xxcon3_device*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+xxcon4_device*(cuf[2]-2.0*cuf[1]+cuf[0])+xxcon5_device*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dx5tx1_device*(ue[3][4]-2.0*ue[2][4]+ue[1][4]);
		/*
		 * ---------------------------------------------------------------------
		 * fourth-order dissipation                         
		 * ---------------------------------------------------------------------
		 */
		if(i==1){
			for(m=0;m<5;m++){forcing(m,i,j,k)=dtemp[m]-dssp_device*(5.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);}
		}else if(i==2){
			for(m=0;m<5;m++){forcing(m,i,j,k)=dtemp[m]-dssp_device*(-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);}
		}else if(i>=3 && i<nx-3){
			for(m=0;m<5;m++){forcing(m,i,j,k)=dtemp[m]-dssp_device*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);}
		}else if(i==nx-3){
			for(m=0;m<5;m++){forcing(m,i,j,k)=dtemp[m]-dssp_device*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]);}
		}else if(i==nx-2){
			for(m=0;m<5;m++){forcing(m,i,j,k)=dtemp[m]-dssp_device*(ue[0][m]-4.0*ue[1][m]+5.0*ue[2][m]);}
		}
		for(m=0;m<5;m++){
			ue[0][m]=ue[1][m]; 
			ue[1][m]=ue[2][m];
			ue[2][m]=ue[3][m];
			ue[3][m]=ue[4][m];
			buf[0][m]=buf[1][m];
			buf[1][m]=buf[2][m];
		}
		cuf[0]=cuf[1];
		cuf[1]=cuf[2];
		q[0]=q[1];
		q[1]=q[2];
		if(i<nx-2){
			dtpp=1.0/ue[3][0];
			for(m=1;m<5;m++){buf[2][m]=dtpp*ue[3][m];}
			cuf[2]=buf[2][1]*buf[2][1];
			buf[2][0]=cuf[2]+buf[2][2]*buf[2][2]+buf[2][3]*buf[2][3];
			q[2]=0.5*(buf[2][1]*ue[3][1]+buf[2][2]*ue[3][2]+buf[2][3]*ue[3][3]);
		}
	}
}

__global__ static void exact_rhs_gpu_kernel_3(double* forcing,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double xi, eta, zeta, dtemp[5], dtpp;
	double ue[5][5], buf[3][5], cuf[3], q[3];

	k=blockIdx.x*blockDim.x+threadIdx.x+1;
	i=blockIdx.y*blockDim.y+threadIdx.y+1;

	if(k>=nz-1 || i>=nx-1){return;}

	zeta=(double)k*dnzm1_device;
	xi=(double)i*dnxm1_device;
	/*
	 * ---------------------------------------------------------------------
	 * eta-direction flux differences             
	 * ---------------------------------------------------------------------
	 */
	for(j=0; j<3; j++){
		eta=(double)j*dnym1_device;
		exact_solution_gpu_device(xi, eta, zeta, dtemp);
		for(m=0;m<5;m++){ue[j+1][m]=dtemp[m];}
		dtpp=1.0/dtemp[0];
		for(m=1;m<5;m++){buf[j][m]=dtpp*dtemp[m];}
		cuf[j]=buf[j][2]*buf[j][2];
		buf[j][0]=cuf[j]+buf[j][1]*buf[j][1]+buf[j][3]*buf[j][3];
		q[j]=0.5*(buf[j][1]*ue[j+1][1]+buf[j][2]*ue[j+1][2]+buf[j][3]*ue[j+1][3]);
	}
	for(j=1; j<ny-1; j++){
		if(j+2<ny){
			eta=(double)(j+2)*dnym1_device;
			exact_solution_gpu_device(xi, eta, zeta, dtemp);
			for(m=0;m<5;m++){ue[4][m]=dtemp[m];}
		}
		dtemp[0]=forcing(0,i,j,k)-ty2_device*(ue[3][2]-ue[1][2])+dy1ty1_device*(ue[3][0]-2.0*ue[2][0]+ue[1][0]);
		dtemp[1]=forcing(1,i,j,k)-ty2_device*(ue[3][1]*buf[2][2]-ue[1][1]*buf[0][2])+yycon2_device*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dy2ty1_device*(ue[3][1]-2.0*ue[2][1]+ ue[1][1]);
		dtemp[2]=forcing(2,i,j,k)-ty2_device*((ue[3][2]*buf[2][2]+c2_device*(ue[3][4]-q[2]))-(ue[1][2]*buf[0][2]+c2_device*(ue[1][4]-q[0])))+yycon1_device*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dy3ty1_device*(ue[3][2]-2.0*ue[2][2]+ue[1][2]);
		dtemp[3]=forcing(3,i,j,k)-ty2_device*(ue[3][3]*buf[2][2]-ue[1][3]*buf[0][2])+yycon2_device*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dy4ty1_device*(ue[3][3]-2.0*ue[2][3]+ue[1][3]);
		dtemp[4]=forcing(4,i,j,k)-ty2_device*(buf[2][2]*(c1_device*ue[3][4]-c2_device*q[2])-buf[0][2]*(c1_device*ue[1][4]-c2_device*q[0]))+0.5*yycon3_device*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+yycon4_device*(cuf[2]-2.0*cuf[1]+cuf[0])+yycon5_device*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dy5ty1_device*(ue[3][4]-2.0*ue[2][4]+ue[1][4]);
		/*
		 * ---------------------------------------------------------------------
		 * fourth-order dissipation                      
		 * ---------------------------------------------------------------------
		 */
		if(j==1){
			for(m=0;m<5;m++){forcing(m,i,j,k)=dtemp[m]-dssp_device*(5.0*ue[2][m]-4.0*ue[3][m] +ue[4][m]);}
		}else if(j==2){
			for(m=0;m<5;m++){forcing(m,i,j,k)=dtemp[m]-dssp_device*(-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);}
		}else if(j>=3 && j<ny-3){
			for(m=0;m<5;m++){forcing(m,i,j,k)=dtemp[m]-dssp_device*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);}
		}else if(j==ny-3){
			for(m=0;m<5;m++){forcing(m,i,j,k)=dtemp[m]-dssp_device*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]);}
		}else if(j==ny-2){
			for(m=0;m<5;m++){forcing(m,i,j,k)=dtemp[m]-dssp_device*(ue[0][m]-4.0*ue[1][m]+5.0*ue[2][m]);}
		}
		for(m=0; m<5; m++){
			ue[0][m]=ue[1][m]; 
			ue[1][m]=ue[2][m];
			ue[2][m]=ue[3][m];
			ue[3][m]=ue[4][m];
			buf[0][m]=buf[1][m];
			buf[1][m]=buf[2][m];
		}
		cuf[0]=cuf[1];
		cuf[1]=cuf[2];
		q[0]=q[1];
		q[1]=q[2];
		if(j<ny-2){
			dtpp=1.0/ue[3][0];
			for(m=1;m<5;m++){buf[2][m]=dtpp*ue[3][m];}
			cuf[2]=buf[2][2]*buf[2][2];
			buf[2][0]=cuf[2]+buf[2][1]*buf[2][1]+buf[2][3]*buf[2][3];
			q[2]=0.5*(buf[2][1]*ue[3][1]+buf[2][2]*ue[3][2]+buf[2][3]*ue[3][3]);
		}
	}
}

__global__ static void exact_rhs_gpu_kernel_4(double* forcing,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double xi, eta, zeta, dtpp, dtemp[5];
	double ue[5][5], buf[3][5], cuf[3], q[3];

	j=blockIdx.x*blockDim.x+threadIdx.x+1;
	i=blockIdx.y*blockDim.y+threadIdx.y+1;

	if(j>=ny-1 || i>=nx-1){return;}

	eta=(double)j*dnym1_device;
	xi=(double)i*dnxm1_device;
	/*
	 * ---------------------------------------------------------------------
	 * zeta-direction flux differences                      
	 * ---------------------------------------------------------------------
	 */
	for(k=0; k<3; k++){
		zeta=(double)k*dnzm1_device;
		exact_solution_gpu_device(xi, eta, zeta, dtemp);
		for(m=0;m<5;m++){ue[k+1][m]=dtemp[m];}
		dtpp=1.0/dtemp[0];
		for(m=1;m<5;m++){buf[k][m]=dtpp*dtemp[m];}
		cuf[k]=buf[k][3]*buf[k][3];
		buf[k][0]=cuf[k]+buf[k][1]*buf[k][1]+buf[k][2]*buf[k][2];
		q[k]=0.5*(buf[k][1]*ue[k+1][1]+buf[k][2]*ue[k+1][2]+buf[k][3]*ue[k+1][3]);
	}
	for(k=1; k<nz-1; k++){
		if(k+2<nz){
			zeta=(double)(k+2)*dnzm1_device;
			exact_solution_gpu_device(xi, eta, zeta, dtemp);
			for(m=0;m<5;m++){ue[4][m]=dtemp[m];}
		}
		dtemp[0]=forcing(0,i,j,k)-tz2_device*(ue[3][3]-ue[1][3])+dz1tz1_device*(ue[3][0]-2.0*ue[2][0]+ue[1][0]);
		dtemp[1]=forcing(1,i,j,k)-tz2_device*(ue[3][1]*buf[2][3]-ue[1][1]*buf[0][3])+zzcon2_device*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dz2tz1_device*(ue[3][1]-2.0*ue[2][1]+ue[1][1]);
		dtemp[2]=forcing(2,i,j,k)-tz2_device*(ue[3][2]*buf[2][3]-ue[1][2]*buf[0][3])+zzcon2_device*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dz3tz1_device*(ue[3][2]-2.0*ue[2][2]+ue[1][2]);
		dtemp[3]=forcing(3,i,j,k)-tz2_device*((ue[3][3]*buf[2][3]+c2_device*(ue[3][4]-q[2]))-(ue[1][3]*buf[0][3]+c2_device*(ue[1][4]-q[0])))+zzcon1_device*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dz4tz1_device*(ue[3][3]-2.0*ue[2][3]+ue[1][3]);
		dtemp[4]=forcing(4,i,j,k)-tz2_device*(buf[2][3]*(c1_device*ue[3][4]-c2_device*q[2])-buf[0][3]*(c1_device*ue[1][4]-c2_device*q[0]))+0.5*zzcon3_device*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+zzcon4_device*(cuf[2]-2.0*cuf[1]+cuf[0])+zzcon5_device*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dz5tz1_device*(ue[3][4]-2.0*ue[2][4]+ue[1][4]);
		/*
		 * ---------------------------------------------------------------------
		 * fourth-order dissipation
		 * ---------------------------------------------------------------------
		 */
		if(k==1){
			for(m=0;m<5;m++){dtemp[m]=dtemp[m]-dssp_device*(5.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);}
		}else if(k==2){
			for(m=0;m<5;m++){dtemp[m]=dtemp[m]-dssp_device*(-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);}
		}else if(k>=3 && k<nz-3){
			for(m=0;m<5;m++){dtemp[m]=dtemp[m]-dssp_device*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);}
		}else if(k==nz-3){
			for(m=0;m<5;m++){dtemp[m]=dtemp[m]-dssp_device*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]);}
		}else if(k==nz-2){
			for(m=0;m<5;m++){dtemp[m]=dtemp[m]-dssp_device*(ue[0][m]-4.0*ue[1][m]+5.0*ue[2][m]);}
		}
		/*
		 * ---------------------------------------------------------------------
		 * now change the sign of the forcing function
		 * ---------------------------------------------------------------------
		 */
		for(m=0;m<5;m++){forcing(m,i,j,k)=-1.0*dtemp[m];}
		for(m=0; m<5; m++){
			ue[0][m]=ue[1][m]; 
			ue[1][m]=ue[2][m];
			ue[2][m]=ue[3][m];
			ue[3][m]=ue[4][m];
			buf[0][m]=buf[1][m];
			buf[1][m]=buf[2][m];
		}
		cuf[0]=cuf[1];
		cuf[1]=cuf[2];
		q[0]=q[1];
		q[1]=q[2];
		if(k<nz-2){
			dtpp=1.0/ue[3][0];
			for(m=1;m<5;m++){buf[2][m]=dtpp*ue[3][m];}
			cuf[2]=buf[2][3]*buf[2][3];
			buf[2][0]=cuf[2]+buf[2][1]*buf[2][1]+buf[2][2]*buf[2][2];
			q[2]=0.5*(buf[2][1]*ue[3][1]+buf[2][2]*ue[3][2]+buf[2][3]*ue[3][3]);
		}
	}
}

/*
 * ---------------------------------------------------------------------
 * this function returns the exact solution at point xi, eta, zeta  
 * ---------------------------------------------------------------------
 */
__device__ static void exact_solution_gpu_device(const double xi,
		const double eta,
		const double zeta,
		double* dtemp){
	for(int m=0; m<5; m++){
		dtemp[m]=ce_device[0][m]+xi*
			(ce_device[1][m]+xi*
			 (ce_device[4][m]+xi*
			  (ce_device[7][m]+xi*
			   ce_device[10][m])))+eta*
			(ce_device[2][m]+eta*
			 (ce_device[5][m]+eta*
			  (ce_device[8][m]+eta*
			   ce_device[11][m])))+zeta*
			(ce_device[3][m]+zeta*
			 (ce_device[6][m]+zeta*
			  (ce_device[9][m]+zeta*
			   ce_device[12][m])));
	}
}

/*
 * ---------------------------------------------------------------------
 * this subroutine initializes the field variable u using 
 * tri-linear transfinite interpolation of the boundary values     
 * ---------------------------------------------------------------------
 */
static void initialize_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_INITIALIZE);
#endif
	/* #KERNEL INITIALIZE */
	int initialize_threads_per_block;
	dim3 initialize_blocks_per_grid(nz, ny);
	if(THREADS_PER_BLOCK_ON_INITIALIZE != nx){
		initialize_threads_per_block = nx;
	}
	else{
		initialize_threads_per_block = THREADS_PER_BLOCK_ON_INITIALIZE;
	}

	initialize_gpu_kernel<<<
		initialize_blocks_per_grid, 
		initialize_threads_per_block>>>(
				u_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_INITIALIZE);
#endif
}

__global__ static void initialize_gpu_kernel(double* u,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double xi, eta, zeta, temp[5];
	double Pface11[5], Pface12[5], Pface21[5], Pface22[5], Pface31[5], Pface32[5];

	k=blockIdx.x;
	j=blockIdx.y;
	i=threadIdx.x;

	/*
	 * ---------------------------------------------------------------------
	 * later (in compute_rhs_gpu) we compute 1/u for every element. a few of 
	 * the corner elements are not used, but it convenient (and faster) 
	 * to compute the whole thing with a simple loop. make sure those 
	 * values are nonzero by initializing the whole thing here. 
	 * ---------------------------------------------------------------------
	 */
	u(0,i,j,k)=1.0;
	u(1,i,j,k)=0.0;
	u(2,i,j,k)=0.0;
	u(3,i,j,k)=0.0;
	u(4,i,j,k)=1.0;
	/*
	 * ---------------------------------------------------------------------
	 * first store the "interpolated" values everywhere on the grid    
	 * ---------------------------------------------------------------------
	 */
	zeta=(double)k*dnzm1_device;
	eta=(double)j*dnym1_device;
	xi=(double)i*dnxm1_device;
	exact_solution_gpu_device(0.0, eta, zeta, Pface11);
	exact_solution_gpu_device(1.0, eta, zeta, Pface12);
	exact_solution_gpu_device(xi, 0.0, zeta, Pface21);
	exact_solution_gpu_device(xi, 1.0, zeta, Pface22);
	exact_solution_gpu_device(xi, eta, 0.0, Pface31);
	exact_solution_gpu_device(xi, eta, 1.0, Pface32);
	for(m=0; m<5; m++){
		double Pxi=xi*Pface12[m]+(1.0-xi)*Pface11[m];
		double Peta=eta*Pface22[m]+(1.0-eta)*Pface21[m];
		double Pzeta=zeta*Pface32[m]+(1.0-zeta)*Pface31[m];
		u(m,i,j,k)=Pxi+Peta+Pzeta-Pxi*Peta-Pxi*Pzeta-Peta*Pzeta+Pxi*Peta*Pzeta;
	}
	/*
	 * ---------------------------------------------------------------------
	 * now store the exact values on the boundaries        
	 * ---------------------------------------------------------------------
	 * west face                                                  
	 * ---------------------------------------------------------------------
	 */
	xi=0.0;
	if(i==0){
		zeta=(double)k*dnzm1_device;
		eta=(double)j*dnym1_device;
		exact_solution_gpu_device(xi, eta, zeta, temp);
		for(m=0;m<5;m++){u(m,i,j,k)=temp[m];}
	}
	/*
	 * ---------------------------------------------------------------------
	 * east face                                                      
	 * ---------------------------------------------------------------------
	 */
	xi=1.0;
	if(i==nx-1){
		zeta=(double)k*dnzm1_device;
		eta=(double)j*dnym1_device;
		exact_solution_gpu_device(xi, eta, zeta, temp);
		for(m=0;m<5;m++){u(m,i,j,k)=temp[m];}
	}
	/*
	 * ---------------------------------------------------------------------
	 * south face                                                 
	 * ---------------------------------------------------------------------
	 */
	eta=0.0;
	if(j==0){
		zeta=(double)k*dnzm1_device;
		xi=(double)i*dnxm1_device;
		exact_solution_gpu_device(xi, eta, zeta, temp);
		for(m=0;m<5;m++){u(m,i,j,k)=temp[m];}
	}
	/*
	 * ---------------------------------------------------------------------
	 * north face                                    
	 * ---------------------------------------------------------------------
	 */
	eta=1.0;
	if(j==ny-1){
		zeta=(double)k*dnzm1_device;
		xi=(double)i*dnxm1_device;
		exact_solution_gpu_device(xi, eta, zeta, temp);
		for(m=0;m<5;m++){u(m,i,j,k)=temp[m];}
	}
	/*
	 * ---------------------------------------------------------------------
	 * bottom face                                       
	 * ---------------------------------------------------------------------
	 */
	zeta=0.0;
	if(k==0){
		eta=(double)j*dnym1_device;
		xi=(double)i*dnxm1_device;
		exact_solution_gpu_device(xi, eta, zeta, temp);
		for(m=0;m<5;m++){u(m,i,j,k)=temp[m];}
	}
	/*
	 * ---------------------------------------------------------------------
	 * top face     
	 * ---------------------------------------------------------------------
	 */
	zeta=1.0;
	if(k==nz-1){
		eta=(double)j*dnym1_device;
		xi=(double)i*dnxm1_device;
		exact_solution_gpu_device(xi, eta, zeta, temp);
		for(m=0;m<5;m++){u(m,i,j,k)=temp[m];}
	}
}

static void release_gpu(){
	hipFree(u_device);
	hipFree(forcing_device);
	hipFree(rhs_device);
	hipFree(rho_i_device);
	hipFree(us_device);
	hipFree(vs_device);
	hipFree(ws_device);
	hipFree(qs_device);
	hipFree(speed_device);
	hipFree(square_device);
	hipFree(lhs_device);
	hipFree(rhs_buffer_device);
	hipFree(rms_buffer_device);
}

static void rhs_norm_gpu(double rms[]){
#if defined(PROFILING)
	timer_start(PROFILING_RHS_NORM_1);
#endif
	/* #KERNEL RHS NORM 1 */
	int rhs_norm_1_threads_per_block = THREADS_PER_BLOCK_ON_RHS_NORM_1;
	dim3 rhs_norm_1_blocks_per_grid(ny, nx);

	rhs_norm_gpu_kernel_1<<<
		rhs_norm_1_blocks_per_grid, 
		rhs_norm_1_threads_per_block>>>(
				rms_buffer_device, 
				rhs_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_RHS_NORM_1);
#endif

#if defined(PROFILING)
	timer_start(PROFILING_RHS_NORM_2);
#endif
	/* #KERNEL RHS NORM 2 */
	int rhs_norm_2_threads_per_block = THREADS_PER_BLOCK_ON_RHS_NORM_2;
	int rhs_norm_2_blocks_per_grid = 1;

	rhs_norm_gpu_kernel_2<<<
		rhs_norm_2_blocks_per_grid,
		rhs_norm_2_threads_per_block,
		sizeof(double)*rhs_norm_2_threads_per_block*5>>>(
				rms_buffer_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_RHS_NORM_2);
#endif

	hipMemcpy(rms, rms_buffer_device, 5*sizeof(double), hipMemcpyDeviceToHost);
}

__global__ static void rhs_norm_gpu_kernel_1(double* rms,
		const double* rhs,
		const int nx,
		const int ny,
		const int nz){
	int i, j, k, m;
	double rms_loc[5];

	j=blockIdx.x*blockDim.x+threadIdx.x;
	i=blockIdx.y*blockDim.y+threadIdx.y;

	if(j>=ny || i>=nx){return;}

	for(m=0;m<5;m++){rms_loc[m]=0.0;}
	if(i>=1 && i<nx-1 && j>=1 && j<ny-1){
		for(k=1; k<nz-1; k++){
			for(int m=0; m<5; m++){
				double add=rhs(m,i,j,k);
				rms_loc[m]+=add*add;
			}
		}
	}
	for(m=0;m<5;m++){rms[i+nx*(j+ny*m)]=rms_loc[m];}
}

__global__ static void rhs_norm_gpu_kernel_2(double* rms,
		const int nx,
		const int ny,
		const int nz){
	int i, m, maxpos, dist;

	double* buffer = (double*)extern_share_data;

	i = threadIdx.x;

	for(m=0;m<5;m++){buffer[i+(m*blockDim.x)]=0.0;}
	while(i<nx*ny){
		for(m=0;m<5;m++){buffer[threadIdx.x+(m*blockDim.x)]+=rms[i+nx*ny*m];}
		i+=blockDim.x;
	}
	maxpos=blockDim.x;
	dist=(maxpos+1)/2;
	i=threadIdx.x;
	__syncthreads();
	while(maxpos>1){
		if(i<dist && i+dist<maxpos){
			for(m=0;m<5;m++){buffer[i+(m*blockDim.x)]+=buffer[(i+dist)+(m*blockDim.x)];}
		}
		maxpos=dist;
		dist=(dist+1)/2;
		__syncthreads();
	}
	m=threadIdx.x;
	if(m<5){rms[m]=sqrt(buffer[0+(m*blockDim.x)]/((double)(nz-2)*(double)(ny-2)*(double)(nx-2)));}
}

static void set_constants(){
	double tx1_host;
	double tx2_host;
	double tx3_host;
	double ty1_host;
	double ty2_host;
	double ty3_host;
	double tz1_host;
	double tz2_host;
	double tz3_host;
	double dx1_host;
	double dx2_host;
	double dx3_host;
	double dx4_host;
	double dx5_host;
	double dy1_host;
	double dy2_host;
	double dy3_host;
	double dy4_host;
	double dy5_host;
	double dz1_host;
	double dz2_host;
	double dz3_host;
	double dz4_host;
	double dz5_host;
	double dssp_host;	
	double dxmax_host;
	double dymax_host;
	double dzmax_host;
	double xxcon1_host;
	double xxcon2_host;
	double xxcon3_host;
	double xxcon4_host;
	double xxcon5_host;
	double dx1tx1_host;
	double dx2tx1_host;
	double dx3tx1_host;
	double dx4tx1_host;
	double dx5tx1_host;
	double yycon1_host;
	double yycon2_host;
	double yycon3_host;
	double yycon4_host;
	double yycon5_host;
	double dy1ty1_host;
	double dy2ty1_host;
	double dy3ty1_host;
	double dy4ty1_host;
	double dy5ty1_host;
	double zzcon1_host;
	double zzcon2_host;
	double zzcon3_host;
	double zzcon4_host;
	double zzcon5_host;
	double dz1tz1_host;
	double dz2tz1_host;
	double dz3tz1_host;
	double dz4tz1_host;
	double dz5tz1_host;
	double dnxm1_host;
	double dnym1_host;
	double dnzm1_host;
	double c1c2_host;
	double c1c5_host;
	double c3c4_host;
	double c1345_host;
	double conz1_host;
	double c1_host;
	double c2_host;
	double c3_host;
	double c4_host;
	double c5_host;
	double c4dssp_host;
	double c5dssp_host;
	double dtdssp_host;
	double dttx1_host;
	double bt_host;
	double dttx2_host;
	double dtty1_host;
	double dtty2_host;
	double dttz1_host;
	double dttz2_host;
	double c2dttx1_host;
	double c2dtty1_host;
	double c2dttz1_host;
	double comz1_host;
	double comz4_host;
	double comz5_host;
	double comz6_host;
	double c3c4tx3_host;
	double c3c4ty3_host;
	double c3c4tz3_host;
	double c2iv_host;
	double con43_host;
	double con16_host;
	
	/* */
	ce_host[0][0]=2.0;
	ce_host[1][0]=0.0;
	ce_host[2][0]=0.0;
	ce_host[3][0]=4.0;
	ce_host[4][0]=5.0;
	ce_host[5][0]=3.0;
	ce_host[6][0]=0.5;
	ce_host[7][0]=0.02;
	ce_host[8][0]=0.01;
	ce_host[9][0]=0.03;
	ce_host[10][0]=0.5;
	ce_host[11][0]=0.4;
	ce_host[12][0]=0.3;
	/* */
	ce_host[0][1]=1.0;
	ce_host[1][1]=0.0;
	ce_host[2][1]=0.0;
	ce_host[3][1]=0.0;
	ce_host[4][1]=1.0;
	ce_host[5][1]=2.0;
	ce_host[6][1]=3.0;
	ce_host[7][1]=0.01;
	ce_host[8][1]=0.03;
	ce_host[9][1]=0.02;
	ce_host[10][1]=0.4;
	ce_host[11][1]=0.3;
	ce_host[12][1]=0.5;
	/* */
	ce_host[0][2]=2.0;
	ce_host[1][2]=2.0;
	ce_host[2][2]=0.0;
	ce_host[3][2]=0.0;
	ce_host[4][2]=0.0;
	ce_host[5][2]=2.0;
	ce_host[6][2]=3.0;
	ce_host[7][2]=0.04;
	ce_host[8][2]=0.03;
	ce_host[9][2]=0.05;
	ce_host[10][2]=0.3;
	ce_host[11][2]=0.5;
	ce_host[12][2]=0.4;
	/* */
	ce_host[0][3]=2.0;
	ce_host[1][3]=2.0;
	ce_host[2][3]=0.0;
	ce_host[3][3]=0.0;
	ce_host[4][3]=0.0;
	ce_host[5][3]=2.0;
	ce_host[6][3]=3.0;
	ce_host[7][3]=0.03;
	ce_host[8][3]=0.05;
	ce_host[9][3]=0.04;
	ce_host[10][3]=0.2;
	ce_host[11][3]=0.1;
	ce_host[12][3]=0.3;
	/* */
	ce_host[0][4]=5.0;
	ce_host[1][4]=4.0;
	ce_host[2][4]=3.0;
	ce_host[3][4]=2.0;
	ce_host[4][4]=0.1;
	ce_host[5][4]=0.4;
	ce_host[6][4]=0.3;
	ce_host[7][4]=0.05;
	ce_host[8][4]=0.04;
	ce_host[9][4]=0.03;
	ce_host[10][4]=0.1;
	ce_host[11][4]=0.3;
	ce_host[12][4]=0.2;
	/* */
	bt_host=sqrt(0.5);
	c1_host=1.4;
	c2_host=0.4;
	c3_host=0.1;
	c4_host=1.0;
	c5_host=1.4;
	dnxm1_host=1.0/(double)(grid_points[0]-1);
	dnym1_host=1.0/(double)(grid_points[1]-1);
	dnzm1_host=1.0/(double)(grid_points[2]-1);
	c1c2_host=c1_host*c2_host;
	c1c5_host=c1_host*c5_host;
	c3c4_host=c3_host*c4_host;
	c1345_host=c1c5_host*c3c4_host;
	conz1_host=(1.0-c1c5_host);
	tx1_host=1.0/(dnxm1_host*dnxm1_host);
	tx2_host=1.0/(2.0*dnxm1_host);
	tx3_host=1.0/dnxm1_host;
	ty1_host=1.0/(dnym1_host*dnym1_host);
	ty2_host=1.0/(2.0*dnym1_host);
	ty3_host=1.0/dnym1_host;
	tz1_host=1.0/(dnzm1_host*dnzm1_host);
	tz2_host=1.0/(2.0*dnzm1_host);
	tz3_host=1.0/dnzm1_host;
	dx1_host=0.75;
	dx2_host=0.75;
	dx3_host=0.75;
	dx4_host=0.75;
	dx5_host=0.75;
	dy1_host=0.75;
	dy2_host=0.75;
	dy3_host=0.75;
	dy4_host=0.75;
	dy5_host=0.75;
	dz1_host=1.0;
	dz2_host=1.0;
	dz3_host=1.0;
	dz4_host=1.0;
	dz5_host=1.0;
	dxmax_host=max(dx3_host, dx4_host);
	dymax_host=max(dy2_host, dy4_host);
	dzmax_host=max(dz2_host, dz3_host);
	dssp_host=0.25*max(dx1_host, max(dy1_host, dz1_host));
	c4dssp_host=4.0*dssp_host;
	c5dssp_host=5.0*dssp_host;
	dttx1_host=dt_host*tx1_host;
	dttx2_host=dt_host*tx2_host;
	dtty1_host=dt_host*ty1_host;
	dtty2_host=dt_host*ty2_host;
	dttz1_host=dt_host*tz1_host;
	dttz2_host=dt_host*tz2_host;
	c2dttx1_host=2.0*dttx1_host;
	c2dtty1_host=2.0*dtty1_host;
	c2dttz1_host=2.0*dttz1_host;
	dtdssp_host=dt_host*dssp_host;
	comz1_host=dtdssp_host;
	comz4_host=4.0*dtdssp_host;
	comz5_host=5.0*dtdssp_host;
	comz6_host=6.0*dtdssp_host;
	c3c4tx3_host=c3c4_host*tx3_host;
	c3c4ty3_host=c3c4_host*ty3_host;
	c3c4tz3_host=c3c4_host*tz3_host;
	dx1tx1_host=dx1_host*tx1_host;
	dx2tx1_host=dx2_host*tx1_host;
	dx3tx1_host=dx3_host*tx1_host;
	dx4tx1_host=dx4_host*tx1_host;
	dx5tx1_host=dx5_host*tx1_host;
	dy1ty1_host=dy1_host*ty1_host;
	dy2ty1_host=dy2_host*ty1_host;
	dy3ty1_host=dy3_host*ty1_host;
	dy4ty1_host=dy4_host*ty1_host;
	dy5ty1_host=dy5_host*ty1_host;
	dz1tz1_host=dz1_host*tz1_host;
	dz2tz1_host=dz2_host*tz1_host;
	dz3tz1_host=dz3_host*tz1_host;
	dz4tz1_host=dz4_host*tz1_host;
	dz5tz1_host=dz5_host*tz1_host;
	c2iv_host=2.5;
	con43_host=4.0/3.0;
	con16_host=1.0/6.0;
	xxcon1_host=c3c4tx3_host*con43_host*tx3_host;
	xxcon2_host=c3c4tx3_host*tx3_host;
	xxcon3_host=c3c4tx3_host*conz1_host*tx3_host;
	xxcon4_host=c3c4tx3_host*con16_host*tx3_host;
	xxcon5_host=c3c4tx3_host*c1c5_host*tx3_host;
	yycon1_host=c3c4ty3_host*con43_host*ty3_host;
	yycon2_host=c3c4ty3_host*ty3_host;
	yycon3_host=c3c4ty3_host*conz1_host*ty3_host;
	yycon4_host=c3c4ty3_host*con16_host*ty3_host;
	yycon5_host=c3c4ty3_host*c1c5_host*ty3_host;
	zzcon1_host=c3c4tz3_host*con43_host*tz3_host;
	zzcon2_host=c3c4tz3_host*tz3_host;
	zzcon3_host=c3c4tz3_host*conz1_host*tz3_host;
	zzcon4_host=c3c4tz3_host*con16_host*tz3_host;
	zzcon5_host=c3c4tz3_host*c1c5_host*tz3_host;
	/* */	
	hipMemcpyToSymbol(HIP_SYMBOL(bt_device), &bt_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c1_device), &c1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c2_device), &c2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c3_device), &c3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c4_device), &c4_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c5_device), &c5_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dnxm1_device), &dnxm1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dnym1_device), &dnym1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dnzm1_device), &dnzm1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c1c2_device), &c1c2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c1c5_device), &c1c5_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c3c4_device), &c3c4_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c1345_device), &c1345_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(conz1_device), &conz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(tx1_device), &tx1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(tx2_device), &tx2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(tx3_device), &tx3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(ty1_device), &ty1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(ty2_device), &ty2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(ty3_device), &ty3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(tz1_device), &tz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(tz2_device), &tz2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(tz3_device), &tz3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dx1_device), &dx1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dx2_device), &dx2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dx3_device), &dx3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dx4_device), &dx4_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dx5_device), &dx5_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dy1_device), &dy1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dy2_device), &dy2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dy3_device), &dy3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dy4_device), &dy4_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dy5_device), &dy5_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dz1_device), &dz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dz2_device), &dz2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dz3_device), &dz3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dz4_device), &dz4_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dz5_device), &dz5_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dxmax_device), &dxmax_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dymax_device), &dymax_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dzmax_device), &dzmax_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dssp_device), &dssp_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c4dssp_device), &c4dssp_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c5dssp_device), &c5dssp_host, sizeof(double));	
	hipMemcpyToSymbol(HIP_SYMBOL(dttx1_device), &dttx1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dttx2_device), &dttx2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dtty1_device), &dtty1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dtty2_device), &dtty2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dttz1_device), &dttz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dttz2_device), &dttz2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c2dttx1_device), &c2dttx1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c2dtty1_device), &c2dtty1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c2dttz1_device), &c2dttz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dtdssp_device), &dtdssp_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(comz1_device), &comz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(comz4_device), &comz4_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(comz5_device), &comz5_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(comz6_device), &comz6_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c3c4tx3_device), &c3c4tx3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c3c4ty3_device), &c3c4ty3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c3c4tz3_device), &c3c4tz3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dx1tx1_device), &dx1tx1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dx2tx1_device), &dx2tx1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dx3tx1_device), &dx3tx1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dx4tx1_device), &dx4tx1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dx5tx1_device), &dx5tx1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dy1ty1_device), &dy1ty1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dy2ty1_device), &dy2ty1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dy3ty1_device), &dy3ty1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dy4ty1_device), &dy4ty1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dy5ty1_device), &dy5ty1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dz1tz1_device), &dz1tz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dz2tz1_device), &dz2tz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dz3tz1_device), &dz3tz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dz4tz1_device), &dz4tz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dz5tz1_device), &dz5tz1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(c2iv_device), &c2iv_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(con43_device), &con43_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(con16_device), &con16_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(xxcon1_device), &xxcon1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(xxcon2_device), &xxcon2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(xxcon3_device), &xxcon3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(xxcon4_device), &xxcon4_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(xxcon5_device), &xxcon5_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(yycon1_device), &yycon1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(yycon2_device), &yycon2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(yycon3_device), &yycon3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(yycon4_device), &yycon4_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(yycon5_device), &yycon5_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(zzcon1_device), &zzcon1_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(zzcon2_device), &zzcon2_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(zzcon3_device), &zzcon3_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(zzcon4_device), &zzcon4_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(zzcon5_device), &zzcon5_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(dt_device), &dt_host, sizeof(double));
	hipMemcpyToSymbol(HIP_SYMBOL(ce_device), &ce_host, 13*5*sizeof(double));
}

static void setup_gpu(){
	/*
	 * struct hipDeviceProp{
	 *  char name[256];
	 *  size_t totalGlobalMem;
	 *  size_t sharedMemPerBlock;
	 *  int regsPerBlock;
	 *  int warpSize;
	 *  size_t memPitch;
	 *  int maxThreadsPerBlock;
	 *  int maxThreadsDim[3];
	 *  int maxGridSize[3];
	 *  size_t totalConstMem;
	 *  int major;
	 *  int minor;
	 *  int clockRate;
	 *  size_t textureAlignment;
	 *  int deviceOverlap;
	 *  int multiProcessorCount;
	 *  int kernelExecTimeoutEnabled;
	 *  int integrated;
	 *  int canMapHostMemory;
	 *  int computeMode;
	 *  int concurrentKernels;
	 *  int ECCEnabled;
	 *  int pciBusID;
	 *  int pciDeviceID;
	 *  int tccDriver;
	 * }
	 */
	/* amount of available devices */ 
	hipGetDeviceCount(&total_devices);

	/* define gpu_device */
	if(total_devices==0){
		printf("\n\n\nNo GPU found!\n\n\n");
		exit(-1);
	}else if((GPU_DEVICE>=0)&&
			(GPU_DEVICE<total_devices)){
		gpu_device_id = GPU_DEVICE;
	}else{
		gpu_device_id = 0;
	}
	hipSetDevice(gpu_device_id);	
	hipGetDeviceProperties(&gpu_device_properties, gpu_device_id);

	/* define threads_per_block */
	if((SP_THREADS_PER_BLOCK_ON_ADD>=1)&&
			(SP_THREADS_PER_BLOCK_ON_ADD<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_ADD = SP_THREADS_PER_BLOCK_ON_ADD;
	}
	else{
		THREADS_PER_BLOCK_ON_ADD = gpu_device_properties.warpSize;
	}
	if((SP_THREADS_PER_BLOCK_ON_COMPUTE_RHS_1>=1)&&
			(SP_THREADS_PER_BLOCK_ON_COMPUTE_RHS_1<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_COMPUTE_RHS_1 = SP_THREADS_PER_BLOCK_ON_COMPUTE_RHS_1;
	}
	else{
		THREADS_PER_BLOCK_ON_COMPUTE_RHS_1 = gpu_device_properties.warpSize;
	}
	if((SP_THREADS_PER_BLOCK_ON_COMPUTE_RHS_2>=1)&&
			(SP_THREADS_PER_BLOCK_ON_COMPUTE_RHS_2<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_COMPUTE_RHS_2 = SP_THREADS_PER_BLOCK_ON_COMPUTE_RHS_2;
	}
	else{
		THREADS_PER_BLOCK_ON_COMPUTE_RHS_2 = gpu_device_properties.warpSize;
	}	
	if((SP_THREADS_PER_BLOCK_ON_ERROR_NORM_1>=1)&&
			(SP_THREADS_PER_BLOCK_ON_ERROR_NORM_1<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_ERROR_NORM_1 = SP_THREADS_PER_BLOCK_ON_ERROR_NORM_1;
	}
	else{
		THREADS_PER_BLOCK_ON_ERROR_NORM_1 = gpu_device_properties.warpSize;
	}
	if((SP_THREADS_PER_BLOCK_ON_ERROR_NORM_2>=1)&&
			(SP_THREADS_PER_BLOCK_ON_ERROR_NORM_2<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_ERROR_NORM_2 = SP_THREADS_PER_BLOCK_ON_ERROR_NORM_2;
	}
	else{
		THREADS_PER_BLOCK_ON_ERROR_NORM_2 = gpu_device_properties.warpSize;
	}
	if((SP_THREADS_PER_BLOCK_ON_EXACT_RHS_1>=1)&&
			(SP_THREADS_PER_BLOCK_ON_EXACT_RHS_1<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_EXACT_RHS_1 = SP_THREADS_PER_BLOCK_ON_EXACT_RHS_1;
	}
	else{
		THREADS_PER_BLOCK_ON_EXACT_RHS_1 = gpu_device_properties.warpSize;
	}
	if((SP_THREADS_PER_BLOCK_ON_EXACT_RHS_2>=1)&&
			(SP_THREADS_PER_BLOCK_ON_EXACT_RHS_2<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_EXACT_RHS_2 = SP_THREADS_PER_BLOCK_ON_EXACT_RHS_2;
	}
	else{
		THREADS_PER_BLOCK_ON_EXACT_RHS_2 = gpu_device_properties.warpSize;
	}
	if((SP_THREADS_PER_BLOCK_ON_EXACT_RHS_3>=1)&&
			(SP_THREADS_PER_BLOCK_ON_EXACT_RHS_3<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_EXACT_RHS_3 = SP_THREADS_PER_BLOCK_ON_EXACT_RHS_3;
	}
	else{
		THREADS_PER_BLOCK_ON_EXACT_RHS_3 = gpu_device_properties.warpSize;
	}
	if((SP_THREADS_PER_BLOCK_ON_EXACT_RHS_4>=1)&&
			(SP_THREADS_PER_BLOCK_ON_EXACT_RHS_4<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_EXACT_RHS_4 = SP_THREADS_PER_BLOCK_ON_EXACT_RHS_4;
	}
	else{
		THREADS_PER_BLOCK_ON_EXACT_RHS_4=gpu_device_properties.warpSize;
	}
	if((SP_THREADS_PER_BLOCK_ON_INITIALIZE>=1)&&
			(SP_THREADS_PER_BLOCK_ON_INITIALIZE<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_INITIALIZE = SP_THREADS_PER_BLOCK_ON_INITIALIZE;
	}
	else{
		THREADS_PER_BLOCK_ON_INITIALIZE=gpu_device_properties.warpSize;
	}
	if((SP_THREADS_PER_BLOCK_ON_RHS_NORM_1>=1)&&
			(SP_THREADS_PER_BLOCK_ON_RHS_NORM_1<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_RHS_NORM_1 = SP_THREADS_PER_BLOCK_ON_RHS_NORM_1;
	}
	else{
		THREADS_PER_BLOCK_ON_RHS_NORM_1 = gpu_device_properties.warpSize;
	}	
	if((SP_THREADS_PER_BLOCK_ON_RHS_NORM_2>=1)&&
			(SP_THREADS_PER_BLOCK_ON_RHS_NORM_2<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_RHS_NORM_2 = SP_THREADS_PER_BLOCK_ON_RHS_NORM_2;
	}
	else{
		THREADS_PER_BLOCK_ON_RHS_NORM_2 = gpu_device_properties.warpSize;
	}	
	if((SP_THREADS_PER_BLOCK_ON_TXINVR>=1)&&
			(SP_THREADS_PER_BLOCK_ON_TXINVR<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_TXINVR = SP_THREADS_PER_BLOCK_ON_TXINVR;
	}
	else{
		THREADS_PER_BLOCK_ON_TXINVR = gpu_device_properties.warpSize;
	}	
	if((SP_THREADS_PER_BLOCK_ON_X_SOLVE>=1)&&
			(SP_THREADS_PER_BLOCK_ON_X_SOLVE<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_X_SOLVE = SP_THREADS_PER_BLOCK_ON_X_SOLVE;
	}
	else{
		THREADS_PER_BLOCK_ON_X_SOLVE = gpu_device_properties.warpSize;
	}	
	if((SP_THREADS_PER_BLOCK_ON_Y_SOLVE>=1)&&
			(SP_THREADS_PER_BLOCK_ON_Y_SOLVE<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_Y_SOLVE = SP_THREADS_PER_BLOCK_ON_Y_SOLVE;
	}
	else{
		THREADS_PER_BLOCK_ON_Y_SOLVE = gpu_device_properties.warpSize;
	}	
	if((SP_THREADS_PER_BLOCK_ON_Z_SOLVE>=1)&&
			(SP_THREADS_PER_BLOCK_ON_Z_SOLVE<=gpu_device_properties.maxThreadsPerBlock)){
		THREADS_PER_BLOCK_ON_Z_SOLVE = SP_THREADS_PER_BLOCK_ON_Z_SOLVE;
	}
	else{
		THREADS_PER_BLOCK_ON_Z_SOLVE = gpu_device_properties.warpSize;
	}	

	int gridsize=nx*ny*nz;
	int facesize=max(max(nx*ny, nx*nz), ny*nz);
	size_u_device=sizeof(double)*(5*gridsize);
	size_forcing_device=sizeof(double)*(5*gridsize);
	size_rhs_device=sizeof(double)*(5*gridsize);
	size_rho_i_device=sizeof(double)*(gridsize);
	size_us_device=sizeof(double)*(gridsize);
	size_vs_device=sizeof(double)*(gridsize);
	size_ws_device=sizeof(double)*(gridsize);
	size_qs_device=sizeof(double)*(gridsize);
	size_speed_device=sizeof(double)*(gridsize);
	size_square_device=sizeof(double)*(gridsize);
	size_lhs_device=sizeof(double)*(9*gridsize);
	size_rhs_buffer_device=sizeof(double)*(5*gridsize);
	size_rms_buffer_device=sizeof(double)*(5*facesize);
	hipMalloc(&u_device, size_u_device);
	hipMalloc(&forcing_device, size_forcing_device);
	hipMalloc(&rhs_device, size_rhs_device);
	hipMalloc(&rho_i_device, size_rho_i_device);
	hipMalloc(&us_device, size_us_device);
	hipMalloc(&vs_device, size_vs_device);
	hipMalloc(&ws_device, size_ws_device);
	hipMalloc(&qs_device, size_qs_device);
	hipMalloc(&speed_device, size_speed_device);
	hipMalloc(&square_device, size_square_device);
	hipMalloc(&lhs_device, size_lhs_device);
	hipMalloc(&rhs_buffer_device, size_rhs_buffer_device);
	hipMalloc(&rms_buffer_device, size_rms_buffer_device);
}

/*
 * ---------------------------------------------------------------------
 * block-diagonal matrix-vector multiplication                  
 * ---------------------------------------------------------------------
 */
static void txinvr_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_TXINVR);
#endif
	/* #KERNEL TXINVR */
	int txinvr_workload = nx * ny * nz;
	int txinvr_threads_per_block = THREADS_PER_BLOCK_ON_TXINVR;
	int txinvr_blocks_per_grid = (ceil((double)txinvr_workload/(double)txinvr_threads_per_block));

	txinvr_gpu_kernel<<<
		txinvr_blocks_per_grid, 
		txinvr_threads_per_block>>>(
				rho_i_device, 
				us_device, 
				vs_device, 
				ws_device, 
				speed_device, 
				qs_device, 
				rhs_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_TXINVR);
#endif
}

__global__ static void txinvr_gpu_kernel(const double* rho_i, 
		const double* us, 
		const double* vs, 
		const double* ws, 
		const double* speed, 
		const double* qs, 
		double* rhs, 
		const int nx, 
		const int ny, 
		const int nz){
	int i_j_k, i, j, k;

	i_j_k = blockIdx.x * blockDim.x + threadIdx.x;

	i = i_j_k % nx;
	j = (i_j_k / nx) % ny;
	k = i_j_k / (nx * ny);

	if(i_j_k >= (nx*ny*nz)){
		return;
	}

	double ru1=rho_i(i,j,k);
	double uu=us(i,j,k);
	double vv=vs(i,j,k);
	double ww=ws(i,j,k);
	double ac=speed(i,j,k);
	double ac2inv=1.0/(ac*ac);
	double r1=rhs(0,i,j,k);
	double r2=rhs(1,i,j,k);
	double r3=rhs(2,i,j,k);
	double r4=rhs(3,i,j,k);
	double r5=rhs(4,i,j,k);
	double t1=c2_device*ac2inv*(qs(i,j,k)*r1-uu*r2-vv*r3-ww*r4+r5);
	double t2=bt_device*ru1*(uu*r1-r2);
	double t3=(bt_device*ru1*ac)*t1;
	rhs(0,i,j,k)=r1-t1;
	rhs(1,i,j,k)=-ru1*(ww*r1-r4);
	rhs(2,i,j,k)=ru1*(vv*r1-r3);
	rhs(3,i,j,k)=-t2+t3;
	rhs(4,i,j,k)=t2+t3;
}

/*
 * ---------------------------------------------------------------------
 * verification routine                         
 * ---------------------------------------------------------------------
 */
static void verify_gpu(int no_time_steps,
		char* class_npb,
		boolean* verified){
	double dt=dt_host;
	double xcrref[5], xceref[5], xcrdif[5], xcedif[5], epsilon, xce[5], xcr[5], dtref;
	int m;
	/*
	 * ---------------------------------------------------------------------
	 * tolerance level
	 * ---------------------------------------------------------------------
	 */
	epsilon=1.0e-08;
	/*
	 * ---------------------------------------------------------------------
	 * compute the error norm and the residual norm, and exit if not printing
	 * ---------------------------------------------------------------------
	 */
	error_norm_gpu(xce);
	compute_rhs_gpu();
	rhs_norm_gpu(xcr);
	for(m=0;m<5;m++){xcr[m]=xcr[m]/dt;}
	*class_npb='U';
	*verified=TRUE;
	for(m=0;m<5;m++){xcrref[m]=1.0;xceref[m]=1.0;}
	/*
	 * ---------------------------------------------------------------------
	 * reference data for 12X12X12 grids after 100 time steps, with DT = 1.50d-02
	 * ---------------------------------------------------------------------
	 */
	if((grid_points[0]==12)&&(grid_points[1]==12)&&(grid_points[2]==12)&&(no_time_steps==100)){
		*class_npb='S';
		dtref=1.5e-2;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=2.7470315451339479e-02;
		xcrref[1]=1.0360746705285417e-02;
		xcrref[2]=1.6235745065095532e-02;
		xcrref[3]=1.5840557224455615e-02;
		xcrref[4]=3.4849040609362460e-02;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=2.7289258557377227e-05;
		xceref[1]=1.0364446640837285e-05;
		xceref[2]=1.6154798287166471e-05;
		xceref[3]=1.5750704994480102e-05;
		xceref[4]=3.4177666183390531e-05;
		/*
		 * ---------------------------------------------------------------------
		 * reference data for 36X36X36 grids after 400 time steps, with DT = 1.5d-03
		 * ---------------------------------------------------------------------
		 */
	}else if((grid_points[0]==36)&&(grid_points[1]==36)&&(grid_points[2]==36)&&(no_time_steps==400)){
		*class_npb='W';
		dtref=1.5e-3;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=0.1893253733584e-02;
		xcrref[1]=0.1717075447775e-03;
		xcrref[2]=0.2778153350936e-03;
		xcrref[3]=0.2887475409984e-03;
		xcrref[4]=0.3143611161242e-02;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=0.7542088599534e-04;
		xceref[1]=0.6512852253086e-05;
		xceref[2]=0.1049092285688e-04;
		xceref[3]=0.1128838671535e-04;
		xceref[4]=0.1212845639773e-03;
		/*
		 * ---------------------------------------------------------------------
		 * reference data for 64X64X64 grids after 400 time steps, with DT = 1.5d-03
		 * ---------------------------------------------------------------------
		 */
	}else if((grid_points[0]==64)&&(grid_points[1]==64)&&(grid_points[2]==64)&&(no_time_steps==400)){
		*class_npb='A';
		dtref=1.5e-3;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual.
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=2.4799822399300195;
		xcrref[1]=1.1276337964368832;
		xcrref[2]=1.5028977888770491;
		xcrref[3]=1.4217816211695179;
		xcrref[4]=2.1292113035138280;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error.
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=1.0900140297820550e-04;
		xceref[1]=3.7343951769282091e-05;
		xceref[2]=5.0092785406541633e-05;
		xceref[3]=4.7671093939528255e-05;
		xceref[4]=1.3621613399213001e-04;
		/*
		 * ---------------------------------------------------------------------
		 * reference data for 102X102X102 grids after 400 time steps,
		 * with DT = 1.0d-03
		 * ---------------------------------------------------------------------
		 */
	}else if((grid_points[0]==102)&&(grid_points[1]==102)&&(grid_points[2]==102)&&(no_time_steps==400)){
		*class_npb='B';
		dtref=1.0e-3;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=0.6903293579998e+02;
		xcrref[1]=0.3095134488084e+02;
		xcrref[2]=0.4103336647017e+02;
		xcrref[3]=0.3864769009604e+02;
		xcrref[4]=0.5643482272596e+02;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=0.9810006190188e-02;
		xceref[1]=0.1022827905670e-02;
		xceref[2]=0.1720597911692e-02;
		xceref[3]=0.1694479428231e-02;
		xceref[4]=0.1847456263981e-01;
		/*
		 * ---------------------------------------------------------------------
		 * reference data for 162X162X162 grids after 400 time steps,
		 * with DT = 0.67d-03
		 * ---------------------------------------------------------------------
		 */
	}else if((grid_points[0]==162)&&(grid_points[1]==162)&&(grid_points[2]==162)&&(no_time_steps==400)){
		*class_npb='C';
		dtref=0.67e-3;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=0.5881691581829e+03;
		xcrref[1]=0.2454417603569e+03;
		xcrref[2]=0.3293829191851e+03;
		xcrref[3]=0.3081924971891e+03;
		xcrref[4]=0.4597223799176e+03;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=0.2598120500183e+00;
		xceref[1]=0.2590888922315e-01;
		xceref[2]=0.5132886416320e-01;
		xceref[3]=0.4806073419454e-01;
		xceref[4]=0.5483377491301e+00;
		/*
		 * ---------------------------------------------------------------------
		 * reference data for 408X408X408 grids after 500 time steps,
		 * with DT = 0.3d-03
		 * ---------------------------------------------------------------------
		 */
	}else if((grid_points[0]==408)&&(grid_points[1]==408)&&(grid_points[2]==408)&&(no_time_steps==500)){
		*class_npb='D';
		dtref=0.30e-3;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=0.1044696216887e+05;
		xcrref[1]=0.3204427762578e+04;
		xcrref[2]=0.4648680733032e+04;
		xcrref[3]=0.4238923283697e+04;
		xcrref[4]=0.7588412036136e+04;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=0.5089471423669e+01;
		xceref[1]=0.5323514855894e+00;
		xceref[2]=0.1187051008971e+01;
		xceref[3]=0.1083734951938e+01;
		xceref[4]=0.1164108338568e+02;
		/*
		 * ---------------------------------------------------------------------
		 * reference data for 1020X1020X1020 grids after 500 time steps,
		 * with DT = 0.1d-03
		 * ---------------------------------------------------------------------
		 */
	}else if((grid_points[0]==1020)&&(grid_points[1]==1020)&&(grid_points[2]==1020)&&(no_time_steps==500)){
		*class_npb='E';
		dtref=0.10e-3;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of residual
		 * ---------------------------------------------------------------------
		 */
		xcrref[0]=0.6255387422609e+05;
		xcrref[1]=0.1495317020012e+05;
		xcrref[2]=0.2347595750586e+05;
		xcrref[3]=0.2091099783534e+05;
		xcrref[4]=0.4770412841218e+05;
		/*
		 * ---------------------------------------------------------------------
		 * reference values of RMS-norms of solution error
		 * ---------------------------------------------------------------------
		 */
		xceref[0]=0.6742735164909e+02;
		xceref[1]=0.5390656036938e+01;
		xceref[2]=0.1680647196477e+02;
		xceref[3]=0.1536963126457e+02;
		xceref[4]=0.1575330146156e+03;
	}else{
		*verified=FALSE;
	}
	/*
	 * ---------------------------------------------------------------------
	 * verification test for residuals if gridsize is one of 
	 * the defined grid sizes above (class .ne. 'U')
	 * ---------------------------------------------------------------------
	 * compute the difference of solution values and the known reference values
	 * ---------------------------------------------------------------------
	 */
	for(m=0; m<5; m++){
		xcrdif[m]=fabs((xcr[m]-xcrref[m])/xcrref[m]);
		xcedif[m]=fabs((xce[m]-xceref[m])/xceref[m]);
	}
	/*
	 * ---------------------------------------------------------------------
	 * output the comparison of computed results to known cases
	 * ---------------------------------------------------------------------
	 */
	if(*class_npb!='U'){
		printf(" Verification being performed for class %c\n",*class_npb);
		printf(" accuracy setting for epsilon = %20.13E\n",epsilon);
		*verified=(fabs(dt-dtref)<=epsilon);
		if(!(*verified)){  
			*class_npb='U';
			printf(" DT does not match the reference value of %15.8E\n",dtref);
		} 
	}else{
		printf(" Unknown class\n");
	}
	if(*class_npb!='U'){
		printf(" Comparison of RMS-norms of residual\n");
	}else{
		printf(" RMS-norms of residual\n");
	}
	for(m=0;m<5;m++){
		if(*class_npb=='U'){
			printf("          %2d%20.13E\n",m+1,xcr[m]);
		}else if(xcrdif[m]<=epsilon){
			printf("          %2d%20.13E%20.13E%20.13E\n",m+1,xcr[m],xcrref[m],xcrdif[m]);
		}else {
			*verified=FALSE;
			printf(" FAILURE: %2d%20.13E%20.13E%20.13E\n",m+1,xcr[m],xcrref[m],xcrdif[m]);
		}
	}
	if(*class_npb!='U'){
		printf(" Comparison of RMS-norms of solution error\n");
	}else{
		printf(" RMS-norms of solution error\n");
	}
	for(m=0;m<5;m++){
		if(*class_npb=='U'){
			printf("          %2d%20.13E\n",m+1,xce[m]);
		}else if(xcedif[m]<=epsilon){
			printf("          %2d%20.13E%20.13E%20.13E\n",m+1,xce[m],xceref[m],xcedif[m]);
		}else{
			*verified = FALSE;
			printf(" FAILURE: %2d%20.13E%20.13E%20.13E\n",m+1,xce[m],xceref[m],xcedif[m]);
		}
	}
	if(*class_npb=='U'){
		printf(" No reference values provided\n");
		printf(" No verification performed\n");
	}else if(*verified){
		printf(" Verification Successful\n");
	}else{
		printf(" Verification failed\n");
	}
}

/*
 * ---------------------------------------------------------------------
 * this function performs the solution of the approximate factorization
 * step in the x-direction for all five matrix components
 * simultaneously. the thomas algorithm is employed to solve the
 * systems for the x-lines. boundary conditions are non-periodic
 * ---------------------------------------------------------------------
 */
static void x_solve_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_X_SOLVE);
#endif
	/* #KERNEL X SOLVE */
	int x_solve_threads_per_block;
	dim3 x_solve_blocks_per_grid(1, nz);
	if(THREADS_PER_BLOCK_ON_X_SOLVE != ny){
		x_solve_threads_per_block = ny;
	}
	else{
		x_solve_threads_per_block = THREADS_PER_BLOCK_ON_X_SOLVE;
	}

	x_solve_gpu_kernel<<<
		x_solve_blocks_per_grid,
		x_solve_threads_per_block>>>(
				rho_i_device, 
				us_device,
				speed_device,
				rhs_device,
				lhs_device,
				rhs_buffer_device,
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_X_SOLVE);
#endif 
}

__global__ static void x_solve_gpu_kernel(const double* rho_i, 
		const double* us, 
		const double* speed, 
		double* rhs, 
		double* lhs, 
		double* rhstmp, 
		const int nx, 
		const int ny, 
		const int nz){
#define lhs(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m-3)))]
#define lhsp(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m+4)))]
#define lhsm(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m-3+2)))]
#define rtmp(m,i,j,k) rhstmp[(j)+ny*((k)+nz*((i)+nx*(m)))]
	int i, j, k, m;
	double rhon[3], cv[3], _lhs[3][5], _lhsp[3][5], _rhs[3][5], fac1;

	/* coalesced */
	j=blockIdx.x*blockDim.x+threadIdx.x+1;
	k=blockIdx.y*blockDim.y+threadIdx.y+1;

	/* uncoalesced */
	/* k=blockIdx.x*blockDim.x+threadIdx.x+1; */
	/* j=blockIdx.y*blockDim.y+threadIdx.y+1; */

	if((k>=nz-1) || (j>=ny-1)){return;}

	/*
	 * ---------------------------------------------------------------------
	 * computes the left hand side for the three x-factors  
	 * ---------------------------------------------------------------------
	 * first fill the lhs for the u-eigenvalue                   
	 * ---------------------------------------------------------------------
	 */
	_lhs[0][0]=lhsp(0,0,j,k)=0.0;
	_lhs[0][1]=lhsp(1,0,j,k)=0.0;
	_lhs[0][2]=lhsp(2,0,j,k)=1.0;
	_lhs[0][3]=lhsp(3,0,j,k)=0.0;
	_lhs[0][4]=lhsp(4,0,j,k)=0.0;
	for(i=0; i<3; i++){
		fac1=c3c4_device*rho_i(i,j,k);
		rhon[i]=max(max(max(dx2_device+con43_device*fac1, dx5_device+c1c5_device*fac1), dxmax_device+fac1), dx1_device);
		cv[i]=us(i,j,k);
	}
	_lhs[1][0]=0.0;
	_lhs[1][1]=-dttx2_device*cv[0]-dttx1_device*rhon[0];
	_lhs[1][2]=1.0+c2dttx1_device*rhon[1];
	_lhs[1][3]=dttx2_device*cv[2]-dttx1_device*rhon[2];
	_lhs[1][4]=0.0;
	_lhs[1][2]+=comz5_device;
	_lhs[1][3]-=comz4_device;
	_lhs[1][4]+=comz1_device;
	for(m=0; m<5; m++){lhsp(m,1,j,k)=_lhs[1][m];}
	rhon[0]=rhon[1];
	rhon[1]=rhon[2];
	cv[0]=cv[1];
	cv[1]=cv[2];
	for(m=0; m<3; m++){
		_rhs[0][m]=rhs(m,0,j,k);
		_rhs[1][m]=rhs(m,1,j,k);
	}
	/*
	 * ---------------------------------------------------------------------
	 * FORWARD ELIMINATION  
	 * ---------------------------------------------------------------------
	 * perform the thomas algorithm; first, FORWARD ELIMINATION     
	 * ---------------------------------------------------------------------
	 */
	for(i=0; i<nx-2; i++){
		/*
		 * ---------------------------------------------------------------------
		 * first fill the lhs for the u-eigenvalue                   
		 * ---------------------------------------------------------------------
		 */
		if((i+2)==(nx-1)){
			_lhs[2][0]=lhsp(0,i+2,j,k)=0.0;
			_lhs[2][1]=lhsp(1,i+2,j,k)=0.0;
			_lhs[2][2]=lhsp(2,i+2,j,k)=1.0;
			_lhs[2][3]=lhsp(3,i+2,j,k)=0.0;
			_lhs[2][4]=lhsp(4,i+2,j,k)=0.0;
		}else{
			fac1=c3c4_device*rho_i(i+3,j,k);
			rhon[2]=max(max(max(dx2_device+con43_device*fac1, dx5_device+c1c5_device*fac1), dxmax_device+fac1), dx1_device);
			cv[2]=us(i+3,j,k);
			_lhs[2][0]=0.0;
			_lhs[2][1]=-dttx2_device*cv[0]-dttx1_device*rhon[0];
			_lhs[2][2]=1.0+c2dttx1_device*rhon[1];
			_lhs[2][3]=dttx2_device*cv[2]-dttx1_device*rhon[2];
			_lhs[2][4]=0.0;
			/*
			 * ---------------------------------------------------------------------
			 * add fourth order dissipation                             
			 * ---------------------------------------------------------------------
			 */
			if((i+2)==(2)){
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz6_device;
				_lhs[2][3]-=comz4_device;
				_lhs[2][4]+=comz1_device;
			}else if((i+2>=3) && (i+2<nx-3)){
				_lhs[2][0]+=comz1_device;
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz6_device;
				_lhs[2][3]-=comz4_device;
				_lhs[2][4]+=comz1_device;
			}else if((i+2)==(nx-3)){
				_lhs[2][0]+=comz1_device;
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz6_device;
				_lhs[2][3]-=comz4_device;
			}else if((i+2)==(nx-2)){
				_lhs[2][0]+=comz1_device;
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz5_device;
			}
			/*
			 * ---------------------------------------------------------------------
			 * store computed lhs for later reuse
			 * ---------------------------------------------------------------------
			 */
			for(m=0;m<5;m++){lhsp(m,i+2,j,k)=_lhs[2][m];}
			rhon[0]=rhon[1];
			rhon[1]=rhon[2];
			cv[0]=cv[1];
			cv[1]=cv[2];
		}
		/*
		 * ---------------------------------------------------------------------
		 * load rhs values for current iteration
		 * ---------------------------------------------------------------------
		 */
		for(m=0;m<3;m++){_rhs[2][m]=rhs(m,i+2,j,k);}
		/*
		 * ---------------------------------------------------------------------
		 * perform current iteration
		 * ---------------------------------------------------------------------
		 */
		fac1=1.0/_lhs[0][2];
		_lhs[0][3]*=fac1;
		_lhs[0][4]*=fac1;
		for(m=0;m<3;m++){_rhs[0][m]*=fac1;}
		_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
		_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
		for(m=0;m<3;m++){_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];}
		_lhs[2][1]-=_lhs[2][0]*_lhs[0][3];
		_lhs[2][2]-=_lhs[2][0]*_lhs[0][4];
		for(m=0;m<3;m++){_rhs[2][m]-=_lhs[2][0]*_rhs[0][m];}
		/*
		 * ---------------------------------------------------------------------
		 * store computed lhs and prepare data for next iteration 
		 * rhs is stored in a temp array such that write accesses are coalesced 
		 * ---------------------------------------------------------------------
		 */
		lhs(3,i,j,k)=_lhs[0][3];
		lhs(4,i,j,k)=_lhs[0][4];
		for(m=0; m<5; m++){
			_lhs[0][m]=_lhs[1][m];
			_lhs[1][m]=_lhs[2][m];
		}
		for(m=0; m<3; m++){
			rtmp(m,i,j,k)=_rhs[0][m];
			_rhs[0][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[2][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * the last two rows in this zone are a bit different,  
	 * since they do not have two more rows available for the
	 * elimination of off-diagonal entries    
	 * ---------------------------------------------------------------------
	 */
	i=nx-2;
	fac1=1.0/_lhs[0][2];
	_lhs[0][3]*=fac1;
	_lhs[0][4]*=fac1;
	for(m=0;m<3;m++){_rhs[0][m]*=fac1;}
	_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
	_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
	for(m=0;m<3;m++){_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];}
	/*
	 * ---------------------------------------------------------------------
	 * scale the last row immediately 
	 * ---------------------------------------------------------------------
	 */
	fac1=1.0/_lhs[1][2];
	for(m=0;m<3;m++){_rhs[1][m]*=fac1;}
	lhs(3,nx-2,j,k)=_lhs[0][3];
	lhs(4,nx-2,j,k)=_lhs[0][4];
	/*
	 * ---------------------------------------------------------------------
	 * subsequently, fill the other factors (u+c), (u-c)
	 * ---------------------------------------------------------------------
	 */
	for(i=0;i<3;i++){cv[i]=speed(i,j,k);}
	for(m=0; m<5; m++){
		_lhsp[0][m]=_lhs[0][m]=lhsp(m,0,j,k);
		_lhsp[1][m]=_lhs[1][m]=lhsp(m,1,j,k);
	}
	_lhsp[1][1]-= dttx2_device*cv[0];
	_lhsp[1][3]+=dttx2_device*cv[2];
	_lhs[1][1]+=dttx2_device*cv[0];
	_lhs[1][3]-=dttx2_device*cv[2];
	cv[0]=cv[1];
	cv[1]=cv[2];
	_rhs[0][3]=rhs(3,0,j,k);
	_rhs[0][4]=rhs(4,0,j,k);
	_rhs[1][3]=rhs(3,1,j,k);
	_rhs[1][4]=rhs(4,1,j,k);
	/*
	 * ---------------------------------------------------------------------
	 * do the u+c and the u-c factors                 
	 * ---------------------------------------------------------------------
	 */
	for(i=0; i<nx-2; i++){
		/*
		 * first, fill the other factors (u+c), (u-c) 
		 * ---------------------------------------------------------------------
		 */
		for(m=0; m<5; m++){
			_lhsp[2][m]=_lhs[2][m]=lhsp(m,i+2,j,k);
		}
		_rhs[2][3]=rhs(3,i+2,j,k);
		_rhs[2][4]=rhs(4,i+2,j,k);
		if((i+2)<(nx-1)){
			cv[2]=speed(i+3,j,k);
			_lhsp[2][1]-=dttx2_device*cv[0];
			_lhsp[2][3]+=dttx2_device*cv[2];
			_lhs[2][1]+=dttx2_device*cv[0];
			_lhs[2][3]-=dttx2_device*cv[2];
			cv[0]=cv[1];
			cv[1]=cv[2];
		}
		m=3;
		fac1=1.0/_lhsp[0][2];
		_lhsp[0][3]*=fac1;
		_lhsp[0][4]*=fac1;
		_rhs[0][m]*=fac1;
		_lhsp[1][2]-=_lhsp[1][1]*_lhsp[0][3];
		_lhsp[1][3]-=_lhsp[1][1]*_lhsp[0][4];
		_rhs[1][m]-=_lhsp[1][1]*_rhs[0][m];
		_lhsp[2][1]-=_lhsp[2][0]*_lhsp[0][3];
		_lhsp[2][2]-=_lhsp[2][0]*_lhsp[0][4];
		_rhs[2][m]-=_lhsp[2][0]*_rhs[0][m];
		m=4;
		fac1=1.0/_lhs[0][2];
		_lhs[0][3]*=fac1;
		_lhs[0][4]*=fac1;
		_rhs[0][m]*=fac1;
		_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
		_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
		_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];
		_lhs[2][1]-=_lhs[2][0]*_lhs[0][3];
		_lhs[2][2]-=_lhs[2][0]*_lhs[0][4];
		_rhs[2][m]-=_lhs[2][0]*_rhs[0][m];
		/*
		 * ---------------------------------------------------------------------
		 * store computed lhs and prepare data for next iteration 
		 * rhs is stored in a temp array such that write accesses are coalesced  
		 * ---------------------------------------------------------------------
		 */
		for(m=3; m<5; m++){
			lhsp(m,i,j,k)=_lhsp[0][m];
			lhsm(m,i,j,k)=_lhs[0][m];
			rtmp(m,i,j,k)=_rhs[0][m];
			_rhs[0][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[2][m];
		}
		for(m=0; m<5; m++){
			_lhsp[0][m]=_lhsp[1][m];
			_lhsp[1][m]=_lhsp[2][m];
			_lhs[0][m]=_lhs[1][m];
			_lhs[1][m]=_lhs[2][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * and again the last two rows separately 
	 * ---------------------------------------------------------------------
	 */
	i=nx-2;
	m=3;
	fac1=1.0/_lhsp[0][2];
	_lhsp[0][3]*=fac1;
	_lhsp[0][4]*=fac1;
	_rhs[0][m]*=fac1;
	_lhsp[1][2]-=_lhsp[1][1]*_lhsp[0][3];
	_lhsp[1][3]-=_lhsp[1][1]*_lhsp[0][4];
	_rhs[1][m]-=_lhsp[1][1]*_rhs[0][m];
	m=4;
	fac1=1.0/_lhs[0][2];
	_lhs[0][3]*=fac1;
	_lhs[0][4]*=fac1;
	_rhs[0][m]*=fac1;
	_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
	_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
	_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];
	/*
	 * ---------------------------------------------------------------------
	 * scale the last row immediately 
	 * ---------------------------------------------------------------------
	 */
	_rhs[1][3]/=_lhsp[1][2];
	_rhs[1][4]/=_lhs[1][2];
	/*
	 * ---------------------------------------------------------------------
	 * BACKSUBSTITUTION 
	 * ---------------------------------------------------------------------
	 */
	for(m=0;m<3;m++){_rhs[0][m]-=lhs(3,nx-2,j,k)*_rhs[1][m];}
	_rhs[0][3]-=_lhsp[0][3]*_rhs[1][3];
	_rhs[0][4]-=_lhs[0][3]*_rhs[1][4];
	for(m=0; m<5; m++){
		_rhs[2][m]=_rhs[1][m];
		_rhs[1][m]=_rhs[0][m];
	}
	for(i=nx-3; i>=0; i--){
		/*
		 * ---------------------------------------------------------------------
		 * the first three factors
		 * ---------------------------------------------------------------------
		 */
		for(m=0; m<3; m++){_rhs[0][m]=rtmp(m,i,j,k)-lhs(3,i,j,k)*_rhs[1][m]-lhs(4,i,j,k)*_rhs[2][m];}
		/*
		 * ---------------------------------------------------------------------
		 * and the remaining two
		 * ---------------------------------------------------------------------
		 */
		_rhs[0][3]=rtmp(3,i,j,k)-lhsp(3,i,j,k)*_rhs[1][3]-lhsp(4,i,j,k)*_rhs[2][3];
		_rhs[0][4]=rtmp(4,i,j,k)-lhsm(3,i,j,k)*_rhs[1][4]-lhsm(4,i,j,k)*_rhs[2][4];
		if(i+2<nx-1){
			/*
			 * ---------------------------------------------------------------------
			 * do the block-diagonal inversion          
			 * ---------------------------------------------------------------------
			 */
			double r1=_rhs[2][0];
			double r2=_rhs[2][1];
			double r3=_rhs[2][2];
			double r4=_rhs[2][3];
			double r5=_rhs[2][4];
			double t1=bt_device*r3;
			double t2=0.5*(r4+r5);
			_rhs[2][0]=-r2;
			_rhs[2][1]=r1;
			_rhs[2][2]=bt_device*(r4-r5);
			_rhs[2][3]=-t1+t2;
			_rhs[2][4]=t1+t2;
		}
		for(m=0; m<5; m++){
			rhs(m,i+2,j,k)=_rhs[2][m];
			_rhs[2][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[0][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * do the block-diagonal inversion          
	 * ---------------------------------------------------------------------
	 */
	double t1=bt_device*_rhs[2][2];
	double t2=0.5*(_rhs[2][3]+_rhs[2][4]);
	rhs(0,1,j,k)=-_rhs[2][1];
	rhs(1,1,j,k)=_rhs[2][0];
	rhs(2,1,j,k)=bt_device*(_rhs[2][3]-_rhs[2][4]);
	rhs(3,1,j,k)=-t1+t2;
	rhs(4,1,j,k)=t1+t2;
	for(m=0;m<5;m++){rhs(m,0,j,k)=_rhs[1][m];}
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp
}

/*
 * ---------------------------------------------------------------------
 * this function performs the solution of the approximate factorization
 * step in the y-direction for all five matrix components
 * simultaneously. the thomas algorithm is employed to solve the
 * systems for the y-lines. boundary conditions are non-periodic
 * ---------------------------------------------------------------------
 */
static void y_solve_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_Y_SOLVE);
#endif
	/* #KERNEL Y SOLVE */
	int y_solve_threads_per_block;
	dim3 y_solve_blocks_per_grid(1, nz);
	if(THREADS_PER_BLOCK_ON_Y_SOLVE != nx){
		y_solve_threads_per_block = nx;
	}
	else{
		y_solve_threads_per_block = THREADS_PER_BLOCK_ON_Y_SOLVE;
	}

	y_solve_gpu_kernel<<<
		y_solve_blocks_per_grid,
		y_solve_threads_per_block>>>(
				rho_i_device, 
				vs_device, 
				speed_device, 
				rhs_device, 
				lhs_device, 
				rhs_buffer_device, 
				nx, 
				ny, 
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_Y_SOLVE);
#endif 
}

__global__ static void y_solve_gpu_kernel(const double* rho_i, 
		const double* vs, 
		const double* speed, 
		double* rhs, 
		double* lhs, 
		double* rhstmp, 
		const int nx, 
		const int ny, 
		const int nz){
#define lhs(m,i,j,k) lhs[(i-1)+(nx-2)*((k-1)+(nz-2)*((j)+ny*(m-3)))]
#define lhsp(m,i,j,k) lhs[(i-1)+(nx-2)*((k-1)+(nz-2)*((j)+ny*(m+4)))]
#define lhsm(m,i,j,k) lhs[(i-1)+(nx-2)*((k-1)+(nz-2)*((j)+ny*(m-3+2)))]
#define rtmp(m,i,j,k) rhstmp[(i)+nx*((k)+nz*((j)+ny*(m)))]
	int i, j, k, m;
	double rhoq[3], cv[3], _lhs[3][5], _lhsp[3][5], _rhs[3][5], fac1;

	/* coalesced */
	i=blockIdx.x*blockDim.x+threadIdx.x+1;
	k=blockIdx.y*blockDim.y+threadIdx.y+1;

	/* uncoalesced */
	/* k=blockIdx.x*blockDim.x+threadIdx.x+1; */
	/* i=blockIdx.y*blockDim.y+threadIdx.y+1; */

	if((k>=(nz-1))||(i>=(nx-1))){return;}

	/*
	 * ---------------------------------------------------------------------
	 * computes the left hand side for the three y-factors   
	 * ---------------------------------------------------------------------
	 * first fill the lhs for the u-eigenvalue         
	 * ---------------------------------------------------------------------
	 */
	_lhs[0][0]=lhsp(0,i,0,k)=0.0;
	_lhs[0][1]=lhsp(1,i,0,k)=0.0;
	_lhs[0][2]=lhsp(2,i,0,k)=1.0;
	_lhs[0][3]=lhsp(3,i,0,k)=0.0;
	_lhs[0][4]=lhsp(4,i,0,k)=0.0;
	for(j=0; j<3; j++){
		fac1=c3c4_device*rho_i(i,j,k);
		rhoq[j]=max(max(max(dy3_device+con43_device*fac1, dy5_device+c1c5_device*fac1), dymax_device+fac1), dy1_device);
		cv[j]=vs(i,j,k);
	}
	_lhs[1][0]=0.0;
	_lhs[1][1]=-dtty2_device*cv[0]-dtty1_device*rhoq[0];
	_lhs[1][2]=1.0+c2dtty1_device*rhoq[1];
	_lhs[1][3]=dtty2_device*cv[2]-dtty1_device*rhoq[2];
	_lhs[1][4]=0.0;
	_lhs[1][2]+=comz5_device;
	_lhs[1][3]-=comz4_device;
	_lhs[1][4]+=comz1_device;
	for(m=0;m<5;m++){lhsp(m,i,1,k)=_lhs[1][m];}
	rhoq[0]=rhoq[1];
	rhoq[1]=rhoq[2];
	cv[0]=cv[1];
	cv[1]=cv[2];
	for(m=0; m<3; m++){
		_rhs[0][m]=rhs(m,i,0,k);
		_rhs[1][m]=rhs(m,i,1,k);
	}
	/*
	 * ---------------------------------------------------------------------
	 * FORWARD ELIMINATION  
	 * ---------------------------------------------------------------------
	 */
	for(j=0; j<ny-2; j++){
		/*
		 * ---------------------------------------------------------------------
		 * first fill the lhs for the u-eigenvalue         
		 * ---------------------------------------------------------------------
		 */
		if((j+2)==(ny-1)){
			_lhs[2][0]=lhsp(0,i,j+2,k)=0.0;
			_lhs[2][1]=lhsp(1,i,j+2,k)=0.0;
			_lhs[2][2]=lhsp(2,i,j+2,k)=1.0;
			_lhs[2][3]=lhsp(3,i,j+2,k)=0.0;
			_lhs[2][4]=lhsp(4,i,j+2,k)=0.0;
		}else{
			fac1=c3c4_device*rho_i(i,j+3,k);
			rhoq[2]=max(max(max(dy3_device+con43_device*fac1, dy5_device+c1c5_device*fac1), dymax_device+fac1), dy1_device);
			cv[2]=vs(i,j+3,k);
			_lhs[2][0]=0.0;
			_lhs[2][1]=-dtty2_device*cv[0]-dtty1_device*rhoq[0];
			_lhs[2][2]=1.0+c2dtty1_device*rhoq[1];
			_lhs[2][3]=dtty2_device*cv[2]-dtty1_device*rhoq[2];
			_lhs[2][4]=0.0;
			/*
			 * ---------------------------------------------------------------------
			 * add fourth order dissipation                             
			 * ---------------------------------------------------------------------
			 */
			if((j+2)==(2)){
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz6_device;
				_lhs[2][3]-=comz4_device;
				_lhs[2][4]+=comz1_device;
			}else if(((j+2)>=(3))&&((j+2)<(ny-3))){
				_lhs[2][0]+=comz1_device;
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz6_device;
				_lhs[2][3]-=comz4_device;
				_lhs[2][4]+=comz1_device;
			}else if((j+2)==(ny-3)){
				_lhs[2][0]+=comz1_device;
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz6_device;
				_lhs[2][3]-=comz4_device;
			}else if((j+2)==(ny-2)){
				_lhs[2][0]+=comz1_device;
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz5_device;
			}
			/*
			 * ---------------------------------------------------------------------
			 * store computed lhs for later reuse                           
			 * ---------------------------------------------------------------------
			 */
			for(m=0;m<5;m++){lhsp(m,i,j+2,k)=_lhs[2][m];}
			rhoq[0]=rhoq[1];
			rhoq[1]=rhoq[2];
			cv[0]=cv[1];
			cv[1]=cv[2];
		}
		/*
		 * ---------------------------------------------------------------------
		 * load rhs values for current iteration                          
		 * ---------------------------------------------------------------------
		 */
		for(m=0;m<3;m++){_rhs[2][m]=rhs(m,i,j+2,k);}
		/*
		 * ---------------------------------------------------------------------
		 * perform current iteration                         
		 * ---------------------------------------------------------------------
		 */
		fac1=1.0/_lhs[0][2];
		_lhs[0][3]*=fac1;
		_lhs[0][4]*=fac1;
		for(m=0;m<3;m++){_rhs[0][m]*=fac1;}
		_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
		_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
		for(m=0;m<3;m++){_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];}
		_lhs[2][1]-=_lhs[2][0]*_lhs[0][3];
		_lhs[2][2]-=_lhs[2][0]*_lhs[0][4];
		for(m=0;m<3;m++){_rhs[2][m]-=_lhs[2][0]*_rhs[0][m];}
		/*
		 * ---------------------------------------------------------------------
		 * store computed lhs and prepare data for next iteration
		 * rhs is stored in a temp array such that write accesses are coalesced                  
		 * ---------------------------------------------------------------------
		 */
		lhs(3,i,j,k)=_lhs[0][3];
		lhs(4,i,j,k)=_lhs[0][4];
		for(m=0; m<5; m++){
			_lhs[0][m]=_lhs[1][m];
			_lhs[1][m]=_lhs[2][m];
		}
		for(m=0; m<3; m++){
			rtmp(m,i,j,k)=_rhs[0][m];
			_rhs[0][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[2][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * the last two rows in this zone are a bit different, 
	 * since they do not have two more rows available for the  
	 * elimination of off-diagonal entries              
	 * ---------------------------------------------------------------------
	 */
	j=ny-2;
	fac1=1.0/_lhs[0][2];
	_lhs[0][3]*=fac1;
	_lhs[0][4]*=fac1;
	for(m=0;m<3;m++){_rhs[0][m]*=fac1;}
	_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
	_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
	for(m=0;m<3;m++){_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];}
	/*
	 * ---------------------------------------------------------------------
	 * scale the last row immediately 
	 * ---------------------------------------------------------------------
	 */
	fac1=1.0/_lhs[1][2];
	for(m=0;m<3;m++){_rhs[1][m]*=fac1;}
	lhs(3,i,ny-2,k)=_lhs[0][3];
	lhs(4,i,ny-2,k)=_lhs[0][4];
	/*
	 * ---------------------------------------------------------------------
	 * do the u+c and the u-c factors                 
	 * ---------------------------------------------------------------------
	 */
	for(j=0;j<3;j++){cv[j]=speed(i,j,k);}
	for(m=0; m<5; m++){
		_lhsp[0][m]=_lhs[0][m]=lhsp(m,i,0,k);
		_lhsp[1][m]=_lhs[1][m]=lhsp(m,i,1,k);
	}
	_lhsp[1][1]-=dtty2_device*cv[0];
	_lhsp[1][3]+=dtty2_device*cv[2];
	_lhs[1][1]+=dtty2_device*cv[0];
	_lhs[1][3]-=dtty2_device*cv[2];
	cv[0]=cv[1];
	cv[1]=cv[2];
	_rhs[0][3]=rhs(3,i,0,k);
	_rhs[0][4]=rhs(4,i,0,k);
	_rhs[1][3]=rhs(3,i,1,k);
	_rhs[1][4]=rhs(4,i,1,k);
	for(j=0; j<ny-2; j++){
		for(m=0; m<5; m++){
			_lhsp[2][m]=_lhs[2][m]=lhsp(m,i,j+2,k);
		}
		_rhs[2][3]=rhs(3,i,j+2,k);
		_rhs[2][4]=rhs(4,i,j+2,k);
		if((j+2)<(ny-1)){
			cv[2]=speed(i,j+3,k);
			_lhsp[2][1]-=dtty2_device*cv[0];
			_lhsp[2][3]+=dtty2_device*cv[2];
			_lhs[2][1]+=dtty2_device*cv[0];
			_lhs[2][3]-=dtty2_device*cv[2];
			cv[0]=cv[1];
			cv[1]=cv[2];
		}
		fac1=1.0/_lhsp[0][2];
		m=3;
		_lhsp[0][3]*=fac1;
		_lhsp[0][4]*=fac1;
		_rhs[0][m]*=fac1;
		_lhsp[1][2]-=_lhsp[1][1]*_lhsp[0][3];
		_lhsp[1][3]-=_lhsp[1][1]*_lhsp[0][4];
		_rhs[1][m]-=_lhsp[1][1]*_rhs[0][m];
		_lhsp[2][1]-=_lhsp[2][0]*_lhsp[0][3];
		_lhsp[2][2]-=_lhsp[2][0]*_lhsp[0][4];
		_rhs[2][m]-=_lhsp[2][0]*_rhs[0][m];
		m=4;
		fac1=1.0/_lhs[0][2];
		_lhs[0][3]*=fac1;
		_lhs[0][4]*=fac1;
		_rhs[0][m]*=fac1;
		_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
		_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
		_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];
		_lhs[2][1]-=_lhs[2][0]*_lhs[0][3];
		_lhs[2][2]-=_lhs[2][0]*_lhs[0][4];
		_rhs[2][m]-=_lhs[2][0]*_rhs[0][m];
		/*
		 * ---------------------------------------------------------------------
		 * store computed lhs and prepare data for next iteration 
		 * rhs is stored in a temp array such that write accesses are coalesced  
		 * ---------------------------------------------------------------------
		 */
		for(m=3; m<5; m++){
			lhsp(m,i,j,k)=_lhsp[0][m];
			lhsm(m,i,j,k)=_lhs[0][m];
			rtmp(m,i,j,k)=_rhs[0][m];
			_rhs[0][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[2][m];
		}
		for(m=0; m<5; m++){
			_lhsp[0][m]=_lhsp[1][m];
			_lhsp[1][m]=_lhsp[2][m];
			_lhs[0][m]=_lhs[1][m];
			_lhs[1][m]=_lhs[2][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * and again the last two rows separately 
	 * ---------------------------------------------------------------------
	 */
	j=ny-2;
	m=3;
	fac1=1.0/_lhsp[0][2];
	_lhsp[0][3]*=fac1;
	_lhsp[0][4]*=fac1;
	_rhs[0][m]*=fac1;
	_lhsp[1][2]-=_lhsp[1][1]*_lhsp[0][3];
	_lhsp[1][3]-=_lhsp[1][1]*_lhsp[0][4];
	_rhs[1][m]-=_lhsp[1][1]*_rhs[0][m];
	m=4;
	fac1=1.0/_lhs[0][2];
	_lhs[0][3]*=fac1;
	_lhs[0][4]*=fac1;
	_rhs[0][m]*=fac1;
	_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
	_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
	_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];
	/*
	 * ---------------------------------------------------------------------
	 * scale the last row immediately 
	 * ---------------------------------------------------------------------
	 */
	_rhs[1][3]/=_lhsp[1][2];
	_rhs[1][4]/=_lhs[1][2];
	/*
	 * ---------------------------------------------------------------------
	 * BACKSUBSTITUTION 
	 * ---------------------------------------------------------------------
	 */
	for(m=0;m<3;m++){_rhs[0][m]-=lhs(3,i,ny-2,k)*_rhs[1][m];}
	_rhs[0][3]-=_lhsp[0][3]*_rhs[1][3];
	_rhs[0][4]-=_lhs[0][3]*_rhs[1][4];
	for(m=0; m<5; m++){
		_rhs[2][m]=_rhs[1][m];
		_rhs[1][m]=_rhs[0][m];
	}
	for(j=ny-3; j>=0; j--){
		/*
		 * ---------------------------------------------------------------------
		 * the first three factors
		 * ---------------------------------------------------------------------
		 */
		for(m=0;m<3;m++){_rhs[0][m]=rtmp(m,i,j,k)-lhs(3,i,j,k)*_rhs[1][m]-lhs(4,i,j,k)*_rhs[2][m];}
		/*
		 * ---------------------------------------------------------------------
		 * and the remaining two
		 * ---------------------------------------------------------------------
		 */
		_rhs[0][3]=rtmp(3,i,j,k)-lhsp(3,i,j,k)*_rhs[1][3]-lhsp(4,i,j,k)*_rhs[2][3];
		_rhs[0][4]=rtmp(4,i,j,k)-lhsm(3,i,j,k)*_rhs[1][4]-lhsm(4,i,j,k)*_rhs[2][4];
		if((j+2)<(ny-1)){
			/*
			 * ---------------------------------------------------------------------
			 * do the block-diagonal inversion          
			 * ---------------------------------------------------------------------
			 */
			double r1=_rhs[2][0];
			double r2=_rhs[2][1];
			double r3=_rhs[2][2];
			double r4=_rhs[2][3];
			double r5=_rhs[2][4];
			double t1=bt_device*r1;
			double t2=0.5*(r4+r5);
			_rhs[2][0]=bt_device*(r4-r5);
			_rhs[2][1]=-r3;
			_rhs[2][2]=r2;
			_rhs[2][3]=-t1+t2;
			_rhs[2][4]=t1+t2;
		}
		for(m=0; m<5; m++){
			rhs(m,i,j+2,k)=_rhs[2][m];
			_rhs[2][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[0][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * do the block-diagonal inversion          
	 * ---------------------------------------------------------------------
	 */
	double t1=bt_device*_rhs[2][0];
	double t2=0.5*(_rhs[2][3]+_rhs[2][4]);
	rhs(0,i,1,k)=bt_device*(_rhs[2][3]-_rhs[2][4]);
	rhs(1,i,1,k)=-_rhs[2][2];
	rhs(2,i,1,k)=_rhs[2][1];
	rhs(3,i,1,k)=-t1+t2;
	rhs(4,i,1,k)=t1+t2;
	for(m=0;m<5;m++){rhs(m,i,0,k)=_rhs[1][m];}
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp
}

/*
 * ---------------------------------------------------------------------
 * this function performs the solution of the approximate factorization
 * step in the z-direction for all five matrix components
 * simultaneously. The Thomas algorithm is employed to solve the
 * systems for the z-lines. Boundary conditions are non-periodic
 * ---------------------------------------------------------------------
 */
static void z_solve_gpu(){
#if defined(PROFILING)
	timer_start(PROFILING_Z_SOLVE);
#endif
	/* #KERNEL Z SOLVE */
	int z_solve_threads_per_block;
	dim3 z_solve_blocks_per_grid(1, ny);
	if(THREADS_PER_BLOCK_ON_Z_SOLVE != nx){
		z_solve_threads_per_block = nx;
	}
	else{
		z_solve_threads_per_block = THREADS_PER_BLOCK_ON_Z_SOLVE;
	}

	z_solve_gpu_kernel<<<
		z_solve_blocks_per_grid, 
		z_solve_threads_per_block>>>(
				rho_i_device,
				us_device,
				vs_device,
				ws_device,
				speed_device,
				qs_device,
				u_device,
				rhs_device,
				lhs_device,
				rhs_buffer_device,
				nx,
				ny,
				nz);
#if defined(PROFILING)
	timer_stop(PROFILING_Z_SOLVE);
#endif 
}

__global__ static void z_solve_gpu_kernel(const double* rho_i,
		const double* us,
		const double* vs,
		const double* ws,
		const double* speed,
		const double* qs,
		const double* u,
		double* rhs,
		double* lhs,
		double* rhstmp,
		const int nx,
		const int ny,
		const int nz){
#define lhs(m,i,j,k) lhs[(i-1)+(nx-2)*((j-1)+(ny-2)*((k)+nz*(m-3)))]
#define lhsp(m,i,j,k) lhs[(i-1)+(nx-2)*((j-1)+(ny-2)*((k)+nz*(m+4)))]
#define lhsm(m,i,j,k) lhs[(i-1)+(nx-2)*((j-1)+(ny-2)*((k)+nz*(m-3+2)))]
#define rtmp(m,i,j,k) rhstmp[(i)+nx*((j)+ny*((k)+nz*(m)))]
	int i, j, k, m;
	double rhos[3], cv[3], _lhs[3][5], _lhsp[3][5], _rhs[3][5], fac1;

	/* coalesced */
	i=blockIdx.x*blockDim.x+threadIdx.x+1;
	j=blockIdx.y*blockDim.y+threadIdx.y+1;

	/* uncoalesced */
	/* j=blockIdx.x*blockDim.x+threadIdx.x+1; */
	/* i=blockIdx.y*blockDim.y+threadIdx.y+1; */

	if((j>=(ny-1))||(i>=(nx-1))){return;}

	/*
	 * ---------------------------------------------------------------------
	 * computes the left hand side for the three z-factors   
	 * ---------------------------------------------------------------------
	 * first fill the lhs for the u-eigenvalue                          
	 * ---------------------------------------------------------------------
	 */
	_lhs[0][0]=lhsp(0,i,j,0)=0.0;
	_lhs[0][1]=lhsp(1,i,j,0)=0.0;
	_lhs[0][2]=lhsp(2,i,j,0)=1.0;
	_lhs[0][3]=lhsp(3,i,j,0)=0.0;
	_lhs[0][4]=lhsp(4,i,j,0)=0.0;
	for(k=0; k<3; k++){
		fac1=c3c4_device*rho_i(i,j,k);
		rhos[k]=max(max(max(dz4_device+con43_device*fac1, dz5_device+c1c5_device*fac1), dzmax_device+fac1), dz1_device);
		cv[k]=ws(i,j,k);
	}
	_lhs[1][0]=0.0;
	_lhs[1][1]=-dttz2_device*cv[0]-dttz1_device*rhos[0];
	_lhs[1][2]=1.0+c2dttz1_device*rhos[1];
	_lhs[1][3]=dttz2_device*cv[2]-dttz1_device*rhos[2];
	_lhs[1][4]=0.0;
	_lhs[1][2]+=comz5_device;
	_lhs[1][3]-=comz4_device;
	_lhs[1][4]+=comz1_device;
	for(m=0; m<5; m++){lhsp(m,i,j,1)=_lhs[1][m];}
	rhos[0]=rhos[1];
	rhos[1]=rhos[2];
	cv[0]=cv[1];
	cv[1]=cv[2];
	for(m=0; m<3; m++){
		_rhs[0][m]=rhs(m,i,j,0);
		_rhs[1][m]=rhs(m,i,j,1);
	}
	/*
	 * ---------------------------------------------------------------------
	 * FORWARD ELIMINATION  
	 * ---------------------------------------------------------------------
	 */
	for(k=0; k<nz-2; k++){
		/*
		 * ---------------------------------------------------------------------
		 * first fill the lhs for the u-eigenvalue                   
		 * ---------------------------------------------------------------------
		 */
		if((k+2)==(nz-1)){
			_lhs[2][0]=lhsp(0,i,j,k+2)=0.0;
			_lhs[2][1]=lhsp(1,i,j,k+2)=0.0;
			_lhs[2][2]=lhsp(2,i,j,k+2)=1.0;
			_lhs[2][3]=lhsp(3,i,j,k+2)=0.0;
			_lhs[2][4]=lhsp(4,i,j,k+2)=0.0;
		}else{
			fac1=c3c4_device*rho_i(i,j,k+3);
			rhos[2]=max(max(max(dz4_device+con43_device*fac1, dz5_device+c1c5_device*fac1), dzmax_device+fac1), dz1_device);
			cv[2]=ws(i,j,k+3);
			_lhs[2][0]=0.0;
			_lhs[2][1]=-dttz2_device*cv[0]-dttz1_device*rhos[0];
			_lhs[2][2]=1.0+c2dttz1_device*rhos[1];
			_lhs[2][3]=dttz2_device*cv[2]-dttz1_device*rhos[2];
			_lhs[2][4]=0.0;
			/*
			 * ---------------------------------------------------------------------
			 * add fourth order dissipation                             
			 * ---------------------------------------------------------------------
			 */
			if((k+2)==(2)){
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz6_device;
				_lhs[2][3]-=comz4_device;
				_lhs[2][4]+=comz1_device;
			}else if(((k+2)>=(3))&&((k+2)<(nz-3))){
				_lhs[2][0]+=comz1_device;
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz6_device;
				_lhs[2][3]-=comz4_device;
				_lhs[2][4]+=comz1_device;
			}else if((k+2)==(nz-3)){
				_lhs[2][0]+=comz1_device;
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz6_device;
				_lhs[2][3]-=comz4_device;
			}else if((k+2)==(nz-2)){
				_lhs[2][0]+=comz1_device;
				_lhs[2][1]-=comz4_device;
				_lhs[2][2]+=comz5_device;
			}
			/*
			 * ---------------------------------------------------------------------
			 * store computed lhs for later reuse
			 * ---------------------------------------------------------------------
			 */
			for(m=0;m<5;m++){lhsp(m,i,j,k+2)=_lhs[2][m];}
			rhos[0]=rhos[1];
			rhos[1]=rhos[2];
			cv[0]=cv[1];
			cv[1]=cv[2];
		}
		/*
		 * ---------------------------------------------------------------------
		 * load rhs values for current iteration
		 * ---------------------------------------------------------------------
		 */
		for(m=0;m<3;m++){_rhs[2][m]=rhs(m,i,j,k+2);}
		/*
		 * ---------------------------------------------------------------------
		 * perform current iteration
		 * ---------------------------------------------------------------------
		 */
		fac1=1.0/_lhs[0][2];
		_lhs[0][3]*=fac1;
		_lhs[0][4]*=fac1;
		for(m=0;m<3;m++){_rhs[0][m]*=fac1;}
		_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
		_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
		for(m=0;m<3;m++){_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];}
		_lhs[2][1]-=_lhs[2][0]*_lhs[0][3];
		_lhs[2][2]-=_lhs[2][0]*_lhs[0][4];
		for(m=0;m<3;m++){_rhs[2][m]-=_lhs[2][0]*_rhs[0][m];}
		/*
		 * ---------------------------------------------------------------------
		 * store computed lhs and prepare data for next iteration 
		 * rhs is stored in a temp array such that write accesses are coalesced 
		 * ---------------------------------------------------------------------
		 */
		lhs(3,i,j,k)=_lhs[0][3];
		lhs(4,i,j,k)=_lhs[0][4];
		for(m=0; m<5; m++){
			_lhs[0][m]=_lhs[1][m];
			_lhs[1][m]=_lhs[2][m];
		}
		for(m=0; m<3; m++){
			rtmp(m,i,j,k)=_rhs[0][m];
			_rhs[0][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[2][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * the last two rows in this zone are a bit different,  
	 * since they do not have two more rows available for the
	 * elimination of off-diagonal entries    
	 * ---------------------------------------------------------------------
	 */
	k=nz-2;
	fac1=1.0/_lhs[0][2];
	_lhs[0][3]*=fac1;
	_lhs[0][4]*=fac1;
	for(m=0;m<3;m++){_rhs[0][m]*=fac1;}
	_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
	_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
	for(m=0;m<3;m++){_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];}
	/*
	 * ---------------------------------------------------------------------
	 * scale the last row immediately 
	 * ---------------------------------------------------------------------
	 */
	fac1=1.0/_lhs[1][2];
	for(m=0;m<3;m++){_rhs[1][m]*=fac1;}
	lhs(3,i,j,k)=_lhs[0][3];
	lhs(4,i,j,k)=_lhs[0][4];
	/*
	 * ---------------------------------------------------------------------
	 * subsequently, fill the other factors (u+c), (u-c)
	 * ---------------------------------------------------------------------
	 */
	for(k=0;k<3;k++){cv[k]=speed(i,j,k);}
	for(m=0;m<5;m++){
		_lhsp[0][m]=_lhs[0][m]=lhsp(m,i,j,0);
		_lhsp[1][m]=_lhs[1][m]=lhsp(m,i,j,1);
	}
	_lhsp[1][1]-=dttz2_device*cv[0];
	_lhsp[1][3]+=dttz2_device*cv[2];
	_lhs[1][1]+=dttz2_device*cv[0];
	_lhs[1][3]-=dttz2_device*cv[2];
	cv[0]=cv[1];
	cv[1]=cv[2];
	_rhs[0][3]=rhs(3,i,j,0);
	_rhs[0][4]=rhs(4,i,j,0);
	_rhs[1][3]=rhs(3,i,j,1);
	_rhs[1][4]=rhs(4,i,j,1);
	/*
	 * ---------------------------------------------------------------------
	 * do the u+c and the u-c factors                 
	 * ---------------------------------------------------------------------
	 */
	for(k=0; k<nz-2; k++){
		/*
		 * first, fill the other factors (u+c), (u-c) 
		 * ---------------------------------------------------------------------
		 */
		for(m=0; m<5; m++){
			_lhsp[2][m]=_lhs[2][m]=lhsp(m,i,j,k+2);
		}
		_rhs[2][3]=rhs(3,i,j,k+2);
		_rhs[2][4]=rhs(4,i,j,k+2);
		if((k+2)<(nz-1)){
			cv[2]=speed(i,j,k+3);
			_lhsp[2][1]-=dttz2_device*cv[0];
			_lhsp[2][3]+=dttz2_device*cv[2];
			_lhs[2][1]+=dttz2_device*cv[0];
			_lhs[2][3]-=dttz2_device*cv[2];
			cv[0]=cv[1];
			cv[1]=cv[2];
		}
		m=3;
		fac1=1.0/_lhsp[0][2];
		_lhsp[0][3]*=fac1;
		_lhsp[0][4]*=fac1;
		_rhs[0][m]*=fac1;
		_lhsp[1][2]-=_lhsp[1][1]*_lhsp[0][3];
		_lhsp[1][3]-=_lhsp[1][1]*_lhsp[0][4];
		_rhs[1][m]-=_lhsp[1][1]*_rhs[0][m];
		_lhsp[2][1]-=_lhsp[2][0]*_lhsp[0][3];
		_lhsp[2][2]-=_lhsp[2][0]*_lhsp[0][4];
		_rhs[2][m]-=_lhsp[2][0]*_rhs[0][m];
		m=4;
		fac1=1.0/_lhs[0][2];
		_lhs[0][3]*= fac1;
		_lhs[0][4]*= fac1;
		_rhs[0][m]*= fac1;
		_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
		_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
		_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];
		_lhs[2][1]-=_lhs[2][0]*_lhs[0][3];
		_lhs[2][2]-=_lhs[2][0]*_lhs[0][4];
		_rhs[2][m]-=_lhs[2][0]*_rhs[0][m];
		/*
		 * ---------------------------------------------------------------------
		 * store computed lhs and prepare data for next iteration 
		 * rhs is stored in a temp array such that write accesses are coalesced  
		 * ---------------------------------------------------------------------
		 */
		for(m=3; m<5; m++){
			lhsp(m,i,j,k)=_lhsp[0][m];
			lhsm(m,i,j,k)=_lhs[0][m];
			rtmp(m,i,j,k)=_rhs[0][m];
			_rhs[0][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[2][m];
		}
		for(m=0; m<5; m++){
			_lhsp[0][m]=_lhsp[1][m];
			_lhsp[1][m]=_lhsp[2][m];
			_lhs[0][m]=_lhs[1][m];
			_lhs[1][m]=_lhs[2][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * and again the last two rows separately 
	 * ---------------------------------------------------------------------
	 */
	k=nz-2;
	m=3;
	fac1=1.0/_lhsp[0][2];
	_lhsp[0][3]*=fac1;
	_lhsp[0][4]*=fac1;
	_rhs[0][m]*=fac1;
	_lhsp[1][2]-=_lhsp[1][1]*_lhsp[0][3];
	_lhsp[1][3]-=_lhsp[1][1]*_lhsp[0][4];
	_rhs[1][m]-=_lhsp[1][1]*_rhs[0][m];
	m=4;
	fac1=1.0/_lhs[0][2];
	_lhs[0][3]*=fac1;
	_lhs[0][4]*=fac1;
	_rhs[0][m]*=fac1;
	_lhs[1][2]-=_lhs[1][1]*_lhs[0][3];
	_lhs[1][3]-=_lhs[1][1]*_lhs[0][4];
	_rhs[1][m]-=_lhs[1][1]*_rhs[0][m];
	/*
	 * ---------------------------------------------------------------------
	 * scale the last row immediately 
	 * ---------------------------------------------------------------------
	 */
	_rhs[1][3]/=_lhsp[1][2];
	_rhs[1][4]/=_lhs[1][2];
	/*
	 * ---------------------------------------------------------------------
	 * BACKSUBSTITUTION 
	 * ---------------------------------------------------------------------
	 */
	for(m=0;m<3;m++){_rhs[0][m]-=lhs(3,i,j,nz-2)*_rhs[1][m];}
	_rhs[0][3]-=_lhsp[0][3]*_rhs[1][3];
	_rhs[0][4]-=_lhs[0][3]*_rhs[1][4];
	for(m=0; m<5; m++){
		_rhs[2][m]=_rhs[1][m];
		_rhs[1][m]=_rhs[0][m];
	}
	for(k=nz-3; k>=0; k--){
		/*
		 * ---------------------------------------------------------------------
		 * the first three factors
		 * ---------------------------------------------------------------------
		 */
		for(m=0;m<3;m++){_rhs[0][m]=rtmp(m,i,j,k)-lhs(3,i,j,k)*_rhs[1][m]-lhs(4,i,j,k)*_rhs[2][m];}
		/*
		 * ---------------------------------------------------------------------
		 * and the remaining two
		 * ---------------------------------------------------------------------
		 */
		_rhs[0][3]=rtmp(3,i,j,k)-lhsp(3,i,j,k)*_rhs[1][3]-lhsp(4,i,j,k)*_rhs[2][3];
		_rhs[0][4]=rtmp(4,i,j,k)-lhsm(3,i,j,k)*_rhs[1][4]-lhsm(4,i,j,k)*_rhs[2][4];
		if((k+2)<(nz-1)){
			/*
			 * ---------------------------------------------------------------------
			 * do the block-diagonal inversion          
			 * ---------------------------------------------------------------------
			 */
			double xvel=us(i,j,k+2);
			double yvel=vs(i,j,k+2);
			double zvel=ws(i,j,k+2);
			double ac=speed(i,j,k+2);
			double uzik1=u(0,i,j,k+2);
			double t1=(bt_device*uzik1)/ac*(_rhs[2][3]+_rhs[2][4]);
			double t2=_rhs[2][2]+t1;
			double t3=bt_device*uzik1*(_rhs[2][3]-_rhs[2][4]);
			_rhs[2][4]=uzik1*(-xvel*_rhs[2][1]+yvel*_rhs[2][0])+qs(i,j,k+2)*t2+c2iv_device*(ac*ac)*t1+zvel*t3;
			_rhs[2][3]=zvel*t2+t3;
			_rhs[2][2]=uzik1*_rhs[2][0]+yvel*t2;
			_rhs[2][1]=-uzik1*_rhs[2][1]+xvel*t2;
			_rhs[2][0]=t2;
		}
		for(m=0; m<5; m++){
			rhs(m,i,j,k+2)=_rhs[2][m];
			_rhs[2][m]=_rhs[1][m];
			_rhs[1][m]=_rhs[0][m];
		}
	}
	/*
	 * ---------------------------------------------------------------------
	 * do the block-diagonal inversion          
	 * ---------------------------------------------------------------------
	 */
	double xvel=us(i,j,1);
	double yvel=vs(i,j,1);
	double zvel=ws(i,j,1);
	double ac=speed(i,j,1);
	double uzik1=u(0,i,j,1);
	double t1=(bt_device*uzik1)/ac*(_rhs[2][3]+_rhs[2][4]);
	double t2=_rhs[2][2]+t1;
	double t3=bt_device*uzik1*(_rhs[2][3]-_rhs[2][4]);
	rhs(4,i,j,1)=uzik1*(-xvel*_rhs[2][1]+yvel*_rhs[2][0])+qs(i,j,1)*t2+c2iv_device*(ac*ac)*t1+zvel*t3;
	rhs(3,i,j,1)=zvel*t2+t3;
	rhs(2,i,j,1)=uzik1*_rhs[2][0]+yvel*t2;
	rhs(1,i,j,1)=-uzik1*_rhs[2][1]+xvel*t2;
	rhs(0,i,j,1)=t2;
	for(m=0;m<5;m++){rhs(m,i,j,0)=_rhs[1][m];}
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp
}
