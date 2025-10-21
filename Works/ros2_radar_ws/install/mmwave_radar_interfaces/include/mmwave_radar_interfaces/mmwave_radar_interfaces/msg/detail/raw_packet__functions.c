// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mmwave_radar_interfaces:msg/RawPacket.idl
// generated code does not contain a copyright notice
#include "mmwave_radar_interfaces/msg/detail/raw_packet__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `data`
#include "std_msgs/msg/detail/byte_multi_array__functions.h"

bool
mmwave_radar_interfaces__msg__RawPacket__init(mmwave_radar_interfaces__msg__RawPacket * msg)
{
  if (!msg) {
    return false;
  }
  // data
  if (!std_msgs__msg__ByteMultiArray__init(&msg->data)) {
    mmwave_radar_interfaces__msg__RawPacket__fini(msg);
    return false;
  }
  return true;
}

void
mmwave_radar_interfaces__msg__RawPacket__fini(mmwave_radar_interfaces__msg__RawPacket * msg)
{
  if (!msg) {
    return;
  }
  // data
  std_msgs__msg__ByteMultiArray__fini(&msg->data);
}

bool
mmwave_radar_interfaces__msg__RawPacket__are_equal(const mmwave_radar_interfaces__msg__RawPacket * lhs, const mmwave_radar_interfaces__msg__RawPacket * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // data
  if (!std_msgs__msg__ByteMultiArray__are_equal(
      &(lhs->data), &(rhs->data)))
  {
    return false;
  }
  return true;
}

bool
mmwave_radar_interfaces__msg__RawPacket__copy(
  const mmwave_radar_interfaces__msg__RawPacket * input,
  mmwave_radar_interfaces__msg__RawPacket * output)
{
  if (!input || !output) {
    return false;
  }
  // data
  if (!std_msgs__msg__ByteMultiArray__copy(
      &(input->data), &(output->data)))
  {
    return false;
  }
  return true;
}

mmwave_radar_interfaces__msg__RawPacket *
mmwave_radar_interfaces__msg__RawPacket__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mmwave_radar_interfaces__msg__RawPacket * msg = (mmwave_radar_interfaces__msg__RawPacket *)allocator.allocate(sizeof(mmwave_radar_interfaces__msg__RawPacket), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mmwave_radar_interfaces__msg__RawPacket));
  bool success = mmwave_radar_interfaces__msg__RawPacket__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mmwave_radar_interfaces__msg__RawPacket__destroy(mmwave_radar_interfaces__msg__RawPacket * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mmwave_radar_interfaces__msg__RawPacket__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mmwave_radar_interfaces__msg__RawPacket__Sequence__init(mmwave_radar_interfaces__msg__RawPacket__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mmwave_radar_interfaces__msg__RawPacket * data = NULL;

  if (size) {
    data = (mmwave_radar_interfaces__msg__RawPacket *)allocator.zero_allocate(size, sizeof(mmwave_radar_interfaces__msg__RawPacket), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mmwave_radar_interfaces__msg__RawPacket__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mmwave_radar_interfaces__msg__RawPacket__fini(&data[i - 1]);
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
mmwave_radar_interfaces__msg__RawPacket__Sequence__fini(mmwave_radar_interfaces__msg__RawPacket__Sequence * array)
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
      mmwave_radar_interfaces__msg__RawPacket__fini(&array->data[i]);
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

mmwave_radar_interfaces__msg__RawPacket__Sequence *
mmwave_radar_interfaces__msg__RawPacket__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mmwave_radar_interfaces__msg__RawPacket__Sequence * array = (mmwave_radar_interfaces__msg__RawPacket__Sequence *)allocator.allocate(sizeof(mmwave_radar_interfaces__msg__RawPacket__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mmwave_radar_interfaces__msg__RawPacket__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mmwave_radar_interfaces__msg__RawPacket__Sequence__destroy(mmwave_radar_interfaces__msg__RawPacket__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mmwave_radar_interfaces__msg__RawPacket__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mmwave_radar_interfaces__msg__RawPacket__Sequence__are_equal(const mmwave_radar_interfaces__msg__RawPacket__Sequence * lhs, const mmwave_radar_interfaces__msg__RawPacket__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mmwave_radar_interfaces__msg__RawPacket__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mmwave_radar_interfaces__msg__RawPacket__Sequence__copy(
  const mmwave_radar_interfaces__msg__RawPacket__Sequence * input,
  mmwave_radar_interfaces__msg__RawPacket__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mmwave_radar_interfaces__msg__RawPacket);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mmwave_radar_interfaces__msg__RawPacket * data =
      (mmwave_radar_interfaces__msg__RawPacket *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mmwave_radar_interfaces__msg__RawPacket__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mmwave_radar_interfaces__msg__RawPacket__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mmwave_radar_interfaces__msg__RawPacket__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
