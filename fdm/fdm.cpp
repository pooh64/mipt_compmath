#include <iostream>
#include <stdio.h>
#include <cmath>
#include <valarray>

typedef union tridiag {
	struct { double a, b, c, d; }; /* input */
	struct { double x, __pad1, __pad2, __pad3; }; /* solution */
	double data[4]; /* raw */
} tridiag_t;

using tridiag_eq = std::valarray<tridiag_t>;

void tridiag_solve(tridiag_eq &eq)
{
	auto const sz = eq.size();
	for (size_t i = 1; i < sz; ++i) {
		auto tmp = eq[i].a / eq[i-1].b;
		eq[i].b -= tmp * eq[i-1].c;
		eq[i].d -= tmp * eq[i-1].d;
	}
	eq[sz-1].x = eq[sz-1].d / eq[sz-1].b;
	for (long i = sz-2; i >= 0; --i)
		eq[i].x = (eq[i].d - eq[i].c * eq[i+1].x) / eq[i].b;
}

inline double func_g (double x) { return 1 + x*x; }
inline double func_Dg(double x) { return 2 * x; }
inline double func_p (double x) { return 2 + cos(x); };
inline double func_f (double x) { return (3+x*x+cos(x))*sin(x) - 2*x*cos(x); }
inline double func_y (double x) { return sin(x); }
constexpr double test_a = std::cos(0);
constexpr double test_b = std::cos(1);
constexpr unsigned test_steps = 20;

void get_test_eq(tridiag_eq *&eq_rp)
{
	size_t sz = test_steps + 2;
	auto &eq = *(eq_rp = new tridiag_eq(sz));
	double inv_h = (test_steps - 1), h = 1.0 / inv_h;
	for (size_t i = 1; i < sz - 1; ++i) {
		double x = i * h;
		double g=func_g(x), Dg=func_Dg(x), p=func_p(x), f=func_f(x);
		eq[i] = {-Dg/2+g*inv_h, -2*g*inv_h-p*h, Dg/2+g*inv_h, -f*h};
	}
	eq[0]    = {0, -1-eq[1].a/eq[1].c, -eq[1].b/eq[1].c,
			2*test_a*h-eq[1].d/eq[1].c};
	eq[sz-1] = {eq[sz-2].b/eq[sz-2].a, 1+eq[sz-2].c/eq[sz-2].a, 0,
			2*test_b*h+eq[sz-2].d/eq[sz-2].a};
}

int main(int argc, char **agrv)
{
	tridiag_eq *eq;
	get_test_eq(eq);
	tridiag_solve(*eq);

	printf("     x;     y(x) calculated;   y(x);\n");
	double max_delta = 0;
	for (size_t i = 1; i < (*eq).size() - 1; ++i) {
		double x = (i - 1) * 1. / (test_steps - 1), calc = (*eq)[i].x;
		printf("%.5e;  %.5e;  %.5e;\n", x, calc, func_y(x));
		double tmp = fabs(calc - func_y(x));
		if (tmp > max_delta)
			max_delta = tmp;
	}
	printf("max_delta = %lg;\n", max_delta);
	return 0;
}
