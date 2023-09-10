#ifndef ARRAY_FUNC_H
#define ARRAY_FUNC_H

int d_(int, int, int);
int m_(int, int, int);
int i_(int, int, int);
int p_ij(int, int, int, int);
int p_o_in_i(int, int, int, int);
int e_i(int, int, int, int);
int e_b(int, int, int, int, int, int);
int e_e(int, int, int, int, int, int);
int g_(int, int, int, int);
double* forward(double*, double*, int*, int, int, int);
double* backward(double*, double*, int*, int, int, int);
double* calc_etas(double*, double*, double*, double*, int*, int, int, int);
double* calc_gammas(double*, int, int);
double calc_log_prob(double*, double*, int*, int, int, int);
void free_mem(double**);

#endif
