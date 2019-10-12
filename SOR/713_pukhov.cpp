#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

struct Vector {
	double *buf;
	std::size_t size;

	inline double &operator[](std::size_t i)
	{
		return buf[i];
	}

	inline double const &operator[](std::size_t i) const
	{
		return buf[i];
	}

	Vector(std::size_t _size)
	{
		size = _size;
		buf = new double[size];
	}

	~Vector()
	{
		delete[] buf;
	}

	void print() const
	{
		for (std::size_t i = 0; i < size; i++)
			std::cout << buf[i] << std::endl;
	}
};

struct Matrix {
	double *buf;
	std::size_t xsize, ysize;

	inline double *operator[](std::size_t y)
	{
		return buf + xsize * y;
	}

	inline double const *operator[](std::size_t y) const
	{
		return buf + xsize * y;
	}

	Matrix(std::size_t _xsize, std::size_t _ysize)
	{
		xsize = _xsize;
		ysize = _ysize;
		buf = new double[xsize * ysize];
	}

	~Matrix()
	{
		delete[] buf;
	}

	void import(std::ifstream &file)
	{
		for (std::size_t i = 0; i < xsize * ysize; ++i)
			file >> buf[i];
	}

	void print() const
	{
		for (std::size_t y = 0; y < ysize; ++y) {
			for (std::size_t x = 0; x < xsize; ++x)
				std::cout << (*this)[y][x] << " ";
			std::cout << std::endl;
		}
	}
};

/* ************************************************************************** */

inline double euclid_norm(Vector const &vec)
{
	double accum = 0;
	for (std::size_t i = 0; i < vec.size; i++)
		accum += vec[i] * vec[i];
	return std::sqrt(accum);
}

/* Do not use operators, put result by reference explicitly
 * => avoid runtime allocations (in constructor)
 */

inline void put_mult(Vector &res, Matrix const &mat, Vector const &vec)
{
	for (std::size_t i = 0; i < mat.ysize; ++i) {
		res[i] = 0;
		for (std::size_t j = 0; j < mat.xsize; ++j) {
			res[i] += mat[i][j] * vec[j];
		}
	}
}

inline void put_mult(Vector &res, double const val, Vector const &vec)
{
	for (std::size_t i = 0; i < vec.size; ++i)
		res[i] = val * vec[i];
}

inline void put_plus(Vector &res, Vector const &v1, Vector const &v2)
{
	for (std::size_t i = 0; i < v1.size; ++i)
		res[i] = v1[i] + v2[i];
}

inline void put_minus(Vector &res, Vector const &v1, Vector const &v2)
{
	for (std::size_t i = 0; i < v1.size; ++i)
		res[i] = v1[i] - v2[i];
}

inline void put_equal(Vector &res, Vector const &v1)
{
	for (std::size_t i = 0; i < v1.size; ++i)
		res[i] = v1[i];
}

/* ************************************************************************** */

Matrix import_sqmat(char const *path)
{
	std::ifstream file;
	file.open(path);

	std::size_t size;
	file >> size;
	Matrix mat(size, size);
	mat.import(file);
	file.close();

	return mat;
}

/* ************************************************************************** */

void seidel_iter(Matrix const &mat, Vector const &vec_f, Vector &sol,
		 Vector &tmp)
{
	std::size_t size = tmp.size;

	for (std::size_t i = 0; i < size; i++) {
		double accum = 0;
		for (std::size_t j = i + 1; j < size; j++)
			accum += mat[i][j] * sol[j];
		tmp[i] = accum;
	}

	put_minus(tmp, vec_f, tmp);

	for (std::size_t i = 0; i < size; i++) {
		double accum = tmp[i];
		for (std::size_t j = 0; j < i; j++)
			accum -= mat[i][j] * sol[j];
		sol[i] = accum / mat[i][i];
	}
}

inline void relax_iter(Matrix const &mat, Vector const &vec_f, Vector &sol,
		       Vector &tmp1, Vector &tmp2, double param)
{
	put_equal(tmp2, sol);			// tmp2 = u(k)
	seidel_iter(mat, vec_f, sol, tmp1);	// sol = z(k+1)

	put_minus(tmp1, sol, tmp2);	// tmp1 = z(k+1) - u(k)
	put_mult(tmp1, param, tmp1);	// tmp1 = w * (z(k+1) - u(k))
	put_plus(sol, tmp2, tmp1);	// u(k+1) = u(k) + w * (z(k+1) - u(k))
}

/* ************************************************************************** */

std::size_t seidel_test(Matrix const &mat, Vector const &vec_f, Vector &sol,
			double residual_norm)
{
	Vector tmp1(vec_f.size);
	std::size_t num = 0;

	for (;; num++) {
		put_mult(tmp1, mat, sol);
		put_minus(tmp1, vec_f, tmp1);
		double norm = euclid_norm(tmp1);
		std::cout << "Iter " << num << ": " << norm << std::endl;

		if (norm <= residual_norm)
			break;

		seidel_iter(mat, vec_f, sol, tmp1);
	}

	return num;
}

std::size_t relax_test(Matrix const &mat, Vector const &vec_f, Vector &sol,
		       double residual_norm, double relax_param)
{
	Vector tmp1(vec_f.size);
	Vector tmp2(vec_f.size);
	std::size_t num = 0;

	for (;; num++) {
		put_mult(tmp1, mat, sol);
		put_minus(tmp1, vec_f, tmp1);
		double norm = euclid_norm(tmp1);
		std::cout << "Iter " << num << ": " << norm << std::endl;

		if (norm <= residual_norm)
			break;

		relax_iter(mat, vec_f, sol, tmp1, tmp2, relax_param);
	}

	return num;
}

/* ************************************************************************** */

int main(int argc, char *argv[])
{
	const double target_residual_norm = 1e-06f;
	const double relax_param = 1.5f;

	if (argc != 2) {
		std::cout << "Missing matrix filename" << std::endl;
		return EXIT_FAILURE;
	}

	Matrix mat = import_sqmat(argv[1]);
	std::size_t size = mat.xsize;
	Vector vec_f(size);
	Vector sol(size);

	for (std::size_t i = 0; i < size; ++i) {
		vec_f[i] = i + 1;
		sol[i] = 0;
	}

	std::cout << "____Seidel method:" << std::endl;
	std::size_t seidel_num = seidel_test(mat, vec_f, sol,
					     target_residual_norm);

	for (std::size_t i = 0; i < size; ++i)
		sol[i] = 0;

	std::cout << "\n____Relaxation method (w = " << relax_param << "):"
		  << std::endl;
	std::size_t relax_num = relax_test(mat, vec_f, sol,
					   target_residual_norm, relax_param);

	std::cout << "\n____Number of iterations:" << std::endl
		  << "\tSeidel: " << seidel_num << std::endl
		  << "\tRelaxation: " << relax_num << std::endl;

	return EXIT_SUCCESS;
}
