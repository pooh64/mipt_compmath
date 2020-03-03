/* c++98 only */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>

struct double2 {
	double data[2];
	double       &operator[](size_t i)       {return data[i];}
	double const &operator[](size_t i) const {return data[i];}
};

inline double2 make_double2(double x, double y)
{
	double2 rv = {x, y};
	return rv;
}

inline double2 operator+(double2 x, double2 y)
{
	return make_double2(x[0] + y[0], x[1] + y[1]);
}

inline double2 operator*(double m, double2 v)
{
	return make_double2(m * v[0], m * v[1]);
}

std::ostream &operator<<(std::ostream &os, double2 v)
{
	os << "(" << v[0] << ", " << v[1] << ")";
}

template <typename V, V f(V), typename H>
inline V euler_iter(V v, H h)
{
	return v + h * f(v);
}

template <typename V, V f(V), typename H>
inline V RK4_iter(V v, H h)
{
	V k1, k2, k3, k4;
	k1 = f(v);
	k2 = f(v + (h / 2.0) * k1);
	k3 = f(v + (h / 2.0) * k2);
	k4 = f(v + h * k3);
	return v + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
}


#define PARAM_L      1.0
#define PARAM_LAMBDA 0.2
#define PARAM_G      3.0

#define PARAM_TIME   3 * (2 * M_PI * sqrt(PARAM_L / PARAM_G))
#define PARAM_STEP   (PARAM_TIME / 30.0)

inline double2 param_func(double2 y)
{
	return make_double2(y[1],
		-1.0 / PARAM_L *(2 * PARAM_LAMBDA * y[1] + PARAM_G * y[0]));
}

inline double param_solution(double t)
{
	double d = PARAM_LAMBDA / PARAM_L;
	double w = PARAM_G / PARAM_L - d * d;
	w = sqrt(w);
	return exp(-d * t) * cos(w * t);
}

inline double2 simple(double2 y)
{
	return make_double2(y[1], -y[0]);
}

int main(int argc, char **argv)
{
	double2 u_start = {1, 0};
	double time = 0, step = PARAM_STEP;

	double2 u_ee, u_rk;
	u_ee = u_start;
	u_rk = u_start;

	while (time <= PARAM_TIME) {
		printf("t:% 8.4f  ee:% 8.4f  rk:% 8.4f  an:% 8.4f\n",
			time, u_ee[0], u_rk[0], param_solution(time));
		u_ee = euler_iter<double2, param_func>(u_ee, step);
		u_rk = RK4_iter  <double2, param_func>(u_rk, step);
		time += step;
	}
	return 0;
}
