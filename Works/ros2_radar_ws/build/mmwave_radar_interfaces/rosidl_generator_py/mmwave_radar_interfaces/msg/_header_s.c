// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from mmwave_radar_interfaces:msg/Header.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "mmwave_radar_interfaces/msg/detail/header__struct.h"
#include "mmwave_radar_interfaces/msg/detail/header__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool mmwave_radar_interfaces__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[43];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("mmwave_radar_interfaces.msg._header.Header", full_classname_dest, 42) == 0);
  }
  mmwave_radar_interfaces__msg__Header * ros_message = _ros_message;
  {  // magic_word
    PyObject * field = PyObject_GetAttrString(_pymsg, "magic_word");
    if (!field) {
      return false;
    }
    {
      // TODO(dirk-thomas) use a better way to check the type before casting
      assert(field->ob_type != NULL);
      assert(field->ob_type->tp_name != NULL);
      assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
      PyArrayObject * seq_field = (PyArrayObject *)field;
      Py_INCREF(seq_field);
      assert(PyArray_NDIM(seq_field) == 1);
      assert(PyArray_TYPE(seq_field) == NPY_UINT8);
      Py_ssize_t size = 8;
      uint8_t * dest = ros_message->magic_word;
      for (Py_ssize_t i = 0; i < size; ++i) {
        uint8_t tmp = *(npy_uint8 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(uint8_t));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }
  {  // version
    PyObject * field = PyObject_GetAttrString(_pymsg, "version");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->version = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // total_packet_len
    PyObject * field = PyObject_GetAttrString(_pymsg, "total_packet_len");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->total_packet_len = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // platform
    PyObject * field = PyObject_GetAttrString(_pymsg, "platform");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->platform = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // frame_number
    PyObject * field = PyObject_GetAttrString(_pymsg, "frame_number");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->frame_number = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // time_cpu_cycles
    PyObject * field = PyObject_GetAttrString(_pymsg, "time_cpu_cycles");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->time_cpu_cycles = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // num_detected_obj
    PyObject * field = PyObject_GetAttrString(_pymsg, "num_detected_obj");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->num_detected_obj = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // num_tlvs
    PyObject * field = PyObject_GetAttrString(_pymsg, "num_tlvs");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->num_tlvs = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // subframe_number
    PyObject * field = PyObject_GetAttrString(_pymsg, "subframe_number");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->subframe_number = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * mmwave_radar_interfaces__msg__header__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of Header */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("mmwave_radar_interfaces.msg._header");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "Header");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  mmwave_radar_interfaces__msg__Header * ros_message = (mmwave_radar_interfaces__msg__Header *)raw_ros_message;
  {  // magic_word
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "magic_word");
    if (!field) {
      return NULL;
    }
    assert(field->ob_type != NULL);
    assert(field->ob_type->tp_name != NULL);
    assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
    PyArrayObject * seq_field = (PyArrayObject *)field;
    assert(PyArray_NDIM(seq_field) == 1);
    assert(PyArray_TYPE(seq_field) == NPY_UINT8);
    assert(sizeof(npy_uint8) == sizeof(uint8_t));
    npy_uint8 * dst = (npy_uint8 *)PyArray_GETPTR1(seq_field, 0);
    uint8_t * src = &(ros_message->magic_word[0]);
    memcpy(dst, src, 8 * sizeof(uint8_t));
    Py_DECREF(field);
  }
  {  // version
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->version);
    {
      int rc = PyObject_SetAttrString(_pymessage, "version", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // total_packet_len
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->total_packet_len);
    {
      int rc = PyObject_SetAttrString(_pymessage, "total_packet_len", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // platform
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->platform);
    {
      int rc = PyObject_SetAttrString(_pymessage, "platform", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // frame_number
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->frame_number);
    {
      int rc = PyObject_SetAttrString(_pymessage, "frame_number", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // time_cpu_cycles
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->time_cpu_cycles);
    {
      int rc = PyObject_SetAttrString(_pymessage, "time_cpu_cycles", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // num_detected_obj
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->num_detected_obj);
    {
      int rc = PyObject_SetAttrString(_pymessage, "num_detected_obj", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // num_tlvs
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->num_tlvs);
    {
      int rc = PyObject_SetAttrString(_pymessage, "num_tlvs", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // subframe_number
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->subframe_number);
    {
      int rc = PyObject_SetAttrString(_pymessage, "subframe_number", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
