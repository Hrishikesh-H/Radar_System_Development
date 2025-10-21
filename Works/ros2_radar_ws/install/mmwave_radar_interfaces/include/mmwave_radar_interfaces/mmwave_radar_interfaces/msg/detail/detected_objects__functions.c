// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mmwave_radar_interfaces:msg/DetectedObjects.idl
// generated code does not contain a copyright notice
#include "mmwave_radar_interfaces/msg/detail/detected_objects__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `x`
// Member `y`
// Member `z`
// Member `velocity`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
mmwave_radar_interfaces__msg__DetectedObjects__init(mmwave_radar_interfaces__msg__DetectedObjects * msg)
{
  if (!msg) {
    return false;
  }
  // num_obj
  // x
  if (!rosidl_runtime_c__float__Sequence__init(&msg->x, 0)) {
    mmwave_radar_interfaces__msg__DetectedObjects__fini(msg);
    return false;
  }
  // y
  if (!rosidl_runtime_c__float__Sequence__init(&msg->y, 0)) {
    mmwave_radar_interfaces__msg__DetectedObjects__fini(msg);
    return false;
  }
  // z
  if (!rosidl_runtime_c__float__Sequence__init(&msg->z, 0)) {
    mmwave_radar_interfaces__msg__DetectedObjects__fini(msg);
    return false;
  }
  // velocity
  if (!rosidl_runtime_c__float__Sequence__init(&msg->velocity, 0)) {
    mmwave_radar_interfaces__msg__DetectedObjects__fini(msg);
    return false;
  }
  return true;
}

void
mmwave_radar_interfaces__msg__DetectedObjects__fini(mmwave_radar_interfaces__msg__DetectedObjects * msg)
{
  if (!msg) {
    return;
  }
  // num_obj
  // x
  rosidl_runtime_c__float__Sequence__fini(&msg->x);
  // y
  rosidl_runtime_c__float__Sequence__fini(&msg->y);
  // z
  rosidl_runtime_c__float__Sequence__fini(&msg->z);
  // velocity
  rosidl_runtime_c__float__Sequence__fini(&msg->velocity);
}

bool
mmwave_radar_interfaces__msg__DetectedObjects__are_equal(const mmwave_radar_interfaces__msg__DetectedObjects * lhs, const mmwave_radar_interfaces__msg__DetectedObjects * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // num_obj
  if (lhs->num_obj != rhs->num_obj) {
    return false;
  }
  // x
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->x), &(rhs->x)))
  {
    return false;
  }
  // y
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->y), &(rhs->y)))
  {
    return false;
  }
  // z
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->z), &(rhs->z)))
  {
    return false;
  }
  // velocity
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->velocity), &(rhs->velocity)))
  {
    return false;
  }
  return true;
}

bool
mmwave_radar_interfaces__msg__DetectedObjects__copy(
  const mmwave_radar_interfaces__msg__DetectedObjects * input,
  mmwave_radar_interfaces__msg__DetectedObjects * output)
{
  if (!input || !output) {
    return false;
  }
  // num_obj
  output->num_obj = input->num_obj;
  // x
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->x), &(output->x)))
  {
    return false;
  }
  // y
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->y), &(output->y)))
  {
    return false;
  }
  // z
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->z), &(output->z)))
  {
    return false;
  }
  // velocity
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->velocity), &(output->velocity)))
  {
    return false;
  }
  return true;
}

mmwave_radar_interfaces__msg__DetectedObjects *
mmwave_radar_interfaces__msg__DetectedObjects__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mmwave_radar_interfaces__msg__DetectedObjects * msg = (mmwave_radar_interfaces__msg__DetectedObjects *)allocator.allocate(sizeof(mmwave_radar_interfaces__msg__DetectedObjects), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mmwave_radar_interfaces__msg__DetectedObjects));
  bool success = mmwave_radar_interfaces__msg__DetectedObjects__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mmwave_radar_interfaces__msg__DetectedObjects__destroy(mmwave_radar_interfaces__msg__DetectedObjects * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mmwave_radar_interfaces__msg__DetectedObjects__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mmwave_radar_interfaces__msg__DetectedObjects__Sequence__init(mmwave_radar_interfaces__msg__DetectedObjects__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mmwave_radar_interfaces__msg__DetectedObjects * data = NULL;

  if (size) {
    data = (mmwave_radar_interfaces__msg__DetectedObjects *)allocator.zero_allocate(size, sizeof(mmwave_radar_interfaces__msg__DetectedObjects), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mmwave_radar_interfaces__msg__DetectedObjects__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mmwave_radar_interfaces__msg__DetectedObjects__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
mmwave_radar_interfaces__msg__DetectedObjects__Sequence__fini(mmwave_radar_interfaces__msg__DetectedObjects__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      mmwave_radar_interfaces__msg__DetectedObjects__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

mmwave_radar_interfaces__msg__DetectedObjects__Sequence *
mmwave_radar_interfaces__msg__DetectedObjects__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mmwave_radar_interfaces__msg__DetectedObjects__Sequence * array = (mmwave_radar_interfaces__msg__DetectedObjects__Sequence *)allocator.allocate(sizeof(mmwave_radar_interfaces__msg__DetectedObjects__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mmwave_radar_interfaces__msg__DetectedObjects__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mmwave_radar_interfaces__msg__DetectedObjects__Sequence__destroy(mmwave_radar_interfaces__msg__DetectedObjects__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mmwave_radar_interfaces__msg__DetectedObjects__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mmwave_radar_interfaces__msg__DetectedObjects__Sequence__are_equal(const mmwave_radar_interfaces__msg__DetectedObjects__Sequence * lhs, const mmwave_radar_interfaces__msg__DetectedObjects__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mmwave_radar_interfaces__msg__DetectedObjects__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mmwave_radar_interfaces__msg__DetectedObjects__Sequence__copy(
  const mmwave_radar_interfaces__msg__DetectedObjects__Sequence * input,
  mmwave_radar_interfaces__msg__DetectedObjects__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mmwave_radar_interfaces__msg__DetectedObjects);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mmwave_radar_interfaces__msg__DetectedObjects * data =
      (mmwave_radar_interfaces__msg__DetectedObjects *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mmwave_radar_interfaces__msg__DetectedObjects__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mmwave_radar_interfaces__msg__DetectedObjects__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mmwave_radar_interfaces__msg__DetectedObjects__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
