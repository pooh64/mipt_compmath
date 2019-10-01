#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <limits>

struct Vector {
	double *buf;
	std::size_t max;

	double &operator[](std::size_t i)
	{
		return buf[i];
	}

	double const &operator[](std::size_t i) const
	{
		return buf[i];
	}

	Vector(std::size_t _max)
	{
		max = _max;
		buf = new double[max];
	}

	~Vector()
	{
		delete[] buf;
	}

	double euc_norm() const
	{
		double accum = 0;
		for (std::size_t i = 0; i < max; i++)
			accum += buf[i] * buf[i];
		return std::sqrt(accum);
	}

	void print() const
	{
		for (std::size_t i = 0; i < max; i++)
			std::cout << buf[i] << std::endl;
	}
};

struct Matrix {
	double *buf;
	std::size_t xmax, ymax;

	double *operator[](std::size_t y)
	{
		return buf + xmax * y;
	}

	double const *operator[](std::size_t y) const
	{
		return buf + xmax * y;
	}

	Matrix(std::size_t _xmax, std::size_t _ymax)
	{
		xmax = _xmax;
		ymax = _ymax;
		buf = new double[xmax * ymax];
	}

	~Matrix()
	{
		delete[] buf;
	}

	void import(std::ifstream &file)
	{
		for (std::size_t i = 0; i < xmax * ymax; ++i)
			file >> buf[i];
	}

	void print() const
	{
		for (std::size_t y = 0; y < ymax; ++y) {
			for (std::size_t x = 0; x < xmax; ++x)
				std::cout << (*this)[y][x] << " ";
			std::cout << std::endl;
		}
	}
};

Matrix get_lu_decomp(Matrix const &mat)
{
	assert(mat.xmax == mat.ymax);
	Matrix lu_mat(mat.xmax, mat.ymax);

	for (std::size_t i = 0; i < mat.xmax; ++i) {
		for (std::size_t j = i; j < mat.xmax; ++j) {
			lu_mat[i][j] = mat[i][j];
			for (std::size_t k = 0; k < i; ++k)
				lu_mat[i][j] -= lu_mat[i][k] * lu_mat[k][j];

			if (j == i)
				continue;

			lu_mat[j][i] = mat[j][i];
			for (std::size_t k = 0; k < i; ++k) {
				if (j == k)
					lu_mat[j][i] -= lu_mat[k][i];
				else
					lu_mat[j][i] -= lu_mat[j][k] *
							lu_mat[k][i];
			}

			lu_mat[j][i] /= lu_mat[i][i];
		}
	}

	return lu_mat;
}

Vector solve_via_lu(Matrix const &mat, Vector const &vec_b)
{
	Matrix lu = get_lu_decomp(mat);
	Vector vec_y(lu.xmax);
	Vector vec_x(lu.ymax);

	for (long i = 0; i < lu.ymax; ++i) {
		double accum = vec_b[i];
		for (long j = 0; j < i; ++j)
			accum -= lu[i][j] * vec_y[j];
		vec_y[i] = accum;
	}

	for (long i = lu.ymax - 1; i >= 0; --i) {
		double accum = vec_y[i];
		for (long j = lu.xmax - 1; j > i; --j)
			accum -= lu[i][j] * vec_x[j];
		vec_x[i] = accum / lu[i][i];
	}

	return vec_x;
}

Vector operator*(Matrix const &mat, Vector &vec)
{
	assert(mat.xmax == vec.max);
	Vector res(mat.ymax);

	for (std::size_t i = 0; i < mat.ymax; ++i) {
		res[i] = 0;
		for (std::size_t j = 0; j < mat.xmax; ++j) {
			res[i] += mat[i][j] * vec[j];
		}
	}
	return res;
}

Vector operator-(Vector const &v1, Vector const &v2)
{
	assert(v1.max == v2.max);
	Vector res(v1.max);
	for (std::size_t i = 0; i < v1.max; i++)
		res[i] = v1[i] - v2[i];
	return res;
}

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

int main(int argc, char *argv[])
{
	if (argc != 2) {
		std::cout << "Missing matrix filename" << std::endl;
	}

	Matrix mat = import_sqmat(argv[1]);

	Vector vec_x(mat.xmax);
	for (std::size_t i = 0; i < vec_x.max; ++i)
		vec_x[i] = i + 1;

	Vector vec_b = mat * vec_x;

	Vector vec_y = solve_via_lu(mat, vec_b);

	Vector vec_delta = vec_y - vec_x;
	std::cout.precision(std::numeric_limits<double>::digits10 + 1);
	std::cout << vec_delta.euc_norm() << std::endl;

	return EXIT_SUCCESS;
}
