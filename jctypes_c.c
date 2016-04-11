/*
This program collects c codes which will be directly called 
from python using ctypes. 
In order to be accessed by ctypes, the code should be compiled as shared library 
gcc jctypes_c.c -shared -o jctypes_c.so
*/

void square(double* array, int n) {
	int ii;
	for( ii = 0; ii < n; ii++) {
		array[ii] = array[ii] * array[ii];
	}
}