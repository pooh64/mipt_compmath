/* c++98 only */
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <limits>
#include <cmath>

struct Vector {
	double *buf;
	size_t size;

	double &operator[](size_t i) { return buf[i]; }
	double const &operator[](size_t i) const { return buf[i]; }

	Vector(size_t _size) { buf = new double[size = _size]; }
	~Vector() { delete[] buf; }

	double L1_norm() const
	{
		double accum = 0;
		for (size_t i = 0; i < size; i++)
			accum += std::abs(buf[i]);
		return accum;
	}

	void print() const
	{
		for (size_t i = 0; i < size; i++)
			std::cout << buf[i] << std::endl;
	}
};

struct Matrix {
	double *buf;
	size_t sizex, sizey;

	double *operator[](size_t y) { return buf + sizex * y; }

	double const *operator[](size_t y) const { return buf + sizex * y; }

	Matrix(size_t _sizex, size_t _sizey)
	{ buf = new double[(sizex = _sizex) * (sizey = _sizey)]; }

	~Matrix() { delete[] buf; }

	void import(std::ifstream &file)
	{
		for (size_t i = 0; i < sizex * sizey; ++i)
			file >> buf[i];
	}

	void print() const
	{
		for (size_t y = 0; y < sizey; ++y) {
			for (size_t x = 0; x < sizex; ++x)
				std::cout << (*this)[y][x] << " ";
			std::cout << std::endl;
		}
	}
};

inline double euclid_norm(Vector const &vec)
{
	double accum = 0;
	for (std::size_t i = 0; i < vec.size; i++)
		accum += vec[i] * vec[i];
	return std::sqrt(accum);
}

inline void put_mult(Vector &res, Matrix const &mat, Vector const &vec)
{
	for (std::size_t i = 0; i < mat.sizey; ++i) {
		res[i] = 0;
		for (std::size_t j = 0; j < mat.sizex; ++j) {
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

void GetLU_Decomp(Matrix const &mat, Matrix &lu_mat)
{
	assert(mat.sizex == mat.sizey);
	assert(lu_mat.sizex == mat.sizex);
	assert(lu_mat.sizey == mat.sizey);

	for (size_t i = 0; i < mat.sizex; ++i) {
		for (size_t j = i; j < mat.sizex; ++j) {
			lu_mat[i][j] = mat[i][j];
			for (size_t k = 0; k < i; ++k)
				lu_mat[i][j] -= lu_mat[i][k] * lu_mat[k][j];

			if (j == i)
				continue;

			lu_mat[j][i] = mat[j][i];
			for (size_t k = 0; k < i; ++k) {
				if (j == k)
					lu_mat[j][i] -= lu_mat[k][i];
				else
					lu_mat[j][i] -=
						lu_mat[j][k] * lu_mat[k][i];
			}

			lu_mat[j][i] /= lu_mat[i][i];
		}
	}
}

struct LU_Solver {
	LU_Solver(size_t size)
		: lu_mat(size, size), vec_y(size)
	{ };

	void operator()(Vector &vec_x, Matrix &mat, Vector &vec_b)
	{
		GetLU_Decomp(mat, lu_mat);

		for (long i = 0; i < lu_mat.sizey; ++i) {
			double accum = vec_b[i];
			for (long j = 0; j < i; ++j)
				accum -= lu_mat[i][j] * vec_y[j];
			vec_y[i] = accum;
		}

		for (long i = lu_mat.sizey - 1; i >= 0; --i) {
			double accum = vec_y[i];
			for (long j = lu_mat.sizex - 1; j > i; --j)
				accum -= lu_mat[i][j] * vec_x[j];
			vec_x[i] = accum / lu_mat[i][i];
		}
	}

private:
	Vector vec_y;
	Matrix lu_mat;
};

/* ************************************************************************** */
// c++98 -> no final/override

struct NewtonFunction {
	virtual void operator()(Vector &res, Vector const &arg) const =0;
};

struct JakobianEval {
	virtual void operator()(Matrix &jk, Vector const &arg) const =0;
};

template <typename _func_t, typename _jk_eval_t>
struct NewtonIterator {
	_func_t func;
	_jk_eval_t jk_eval;

	NewtonIterator(size_t size) :
		lu_solver(size), jk(size, size), right(size)
	{ }

	void operator()(Vector &next, Vector const &prev)
	{
		func(right, prev);
		jk_eval(jk, prev);
		lu_solver(next, jk, right);
		put_minus(next, prev, next);
	}

private:
	LU_Solver lu_solver;
	Vector right;
	Matrix jk;
};


struct TestVecFunc: NewtonFunction {
	Matrix const *mat;

	void operator()(Vector &res, Vector const &arg) const
	{
		put_mult(res, *mat, arg);
		for (size_t i = 0; i < res.size; ++i)
			res[i] -= std::exp(-arg[i]);
	}
};

struct TestJakobianEval: JakobianEval {
	Matrix const *mat;

	void operator()(Matrix &jk, Vector const &arg) const
	{
		assert(jk.sizex == mat->sizex);
		assert(jk.sizey == mat->sizey);
		for (size_t i = 0; i < jk.sizey; ++i) {
			for (size_t j = 0; j < jk.sizex; ++j) {
				jk[i][j] = (*mat)[i][j];
				if (i == j)
					jk[i][j] += std::exp(-arg[i]);
			}
		}
	}
};

#define TARGET_RES 1e-06

int main(int argc, char *argv[])
{
	if (argc != 2) {
		std::cout << "Missing matrix filename" << std::endl;
		return EXIT_FAILURE;
	}
	Matrix mat = import_sqmat(argv[1]);
	size_t size = mat.sizex;

	NewtonIterator<TestVecFunc, TestJakobianEval> newt_iter(size);
	newt_iter.func.mat = &mat;
	newt_iter.jk_eval.mat = &mat;

	Vector vec_x(size);
	Vector next(size);
	for (size_t i = 0; i < size; ++i)
		vec_x[i] = 0;

	while (1) {
		newt_iter(next, vec_x);

		for (size_t i = 0; i < next.size; ++i)
			std::cout << next[i] << " ";
		std::cout << std::endl;
		put_minus(vec_x, next, vec_x);
		double res = euclid_norm(vec_x);
		std::cout << "res: " << res << std::endl;

		put_equal(vec_x, next);

		if (res <= TARGET_RES)
			break;
	}

	return EXIT_SUCCESS;
}
