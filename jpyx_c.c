#include <stdio.h>

#include "Python.h"

#include "jc.h"

//In most cases, the easiest way to deal with this problem is to rewrite your C source 
//to use Pythonic methods, e.g. PySys_WriteStdout:
//https://github.com/ipython/ipython/issues/1230 
#define printf PySys_WriteStdout

// In C int foo() and int foo(void) are different functions.
// http://stackoverflow.com/questions/42125/function-declaration-isnt-a-prototype 
int prt( void)
{
	printf( "Hello\n");

	return 0;
}

int prt_str( char* str)
{
	printf( "%s\n",  str);

	return 0;
}

unsigned int bin_sum( unsigned int a, unsigned int b)
{
	unsigned int c;
	c = a + b;

	return c;
}

int sumup( int N) {
    int s = 0;
    int ii, jj, kk;

    for( ii = 0; ii < N; ii++) {
        for( jj = 0; jj < N; jj++) {
            for( kk = 0; kk < N; kk++) {
                s += ii * jj * kk;
            }
        }
    }
    return s;
}