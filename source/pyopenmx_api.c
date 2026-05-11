#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "openmx_api.h"

static PyObject *py_run(PyObject *self, PyObject *args)
{
  long long fcomm_ll;
  const char *input_file;
  int nth;
  int ierr;

  if (!PyArg_ParseTuple(args, "Lsi", &fcomm_ll, &input_file, &nth)) {
    return NULL;
  }

  ierr = openmx_api_f((MPI_Fint)fcomm_ll, input_file, nth);

  return PyLong_FromLong((long)ierr);
}

static PyMethodDef PyOpenMXMethods[] = {
  {"run", py_run, METH_VARARGS, "Run OpenMX via openmx_api_f(fcomm, input_file, nth)."},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pyopenmx_apimodule = {
  PyModuleDef_HEAD_INIT,
  "pyopenmx_api",
  NULL,
  -1,
  PyOpenMXMethods
};

PyMODINIT_FUNC PyInit_pyopenmx_api(void)
{
  return PyModule_Create(&pyopenmx_apimodule);
}
