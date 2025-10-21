// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from mmwave_radar_interfaces:msg/Header.idl
// generated code does not contain a copyright notice
#include "mmwave_radar_interfaces/msg/detail/header__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
mmwave_radar_interfaces__msg__Header__init(mmwave_radar_interfaces__msg__Header * msg)
{
  if (!msg) {
    return false;
  }
  // magic_word
  // version
  // total_packet_len
  // platform
  // frame_number
  // time_cpu_cycles
  // num_detected_obj
  // num_tlvs
  // subframe_number
  return true;
}

void
mmwave_radar_interfaces__msg__Header__fini(mmwave_radar_interfaces__msg__Header * msg)
{
  if (!msg) {
    return;
  }
  // magic_word
  // version
  // total_packet_len
  // platform
  // frame_number
  // time_cpu_cycles
  // num_detected_obj
  // num_tlvs
  // subframe_number
}

bool
mmwave_radar_interfaces__msg__Header__are_equal(const mmwave_radar_interfaces__msg__Header * lhs, const mmwave_radar_interfaces__msg__Header * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // magic_word
  for (size_t i = 0; i < 8; ++i) {
    if (lhs->magic_word[i] != rhs->magic_word[i]) {
      return false;
    }
  }
  // version
  if (lhs->version != rhs->version) {
    return false;
  }
  // total_packet_len
  if (lhs->total_packet_len != rhs->total_packet_len) {
    return false;
  }
  // platform
  if (lhs->platform != rhs->platform) {
    return false;
  }
  // frame_number
  if (lhs->frame_number != rhs->frame_number) {
    return false;
  }
  // time_cpu_cycles
  if (lhs->time_cpu_cycles != rhs->time_cpu_cycles) {
    return false;
  }
  // num_detected_obj
  if (lhs->num_detected_obj != rhs->num_detected_obj) {
    return false;
  }
  // num_tlvs
  if (lhs->num_tlvs != rhs->num_tlvs) {
    return false;
  }
  // subframe_number
  if (lhs->subframe_number != rhs->subframe_number) {
    return false;
  }
  return true;
}

bool
mmwave_radar_interfaces__msg__Header__copy(
  const mmwave_radar_interfaces__msg__Header * input,
  mmwave_radar_interfaces__msg__Header * output)
{
  if (!input || !output) {
    return false;
  }
  // magic_word
  for (size_t i = 0; i < 8; ++i) {
    output->magic_word[i] = input->magic_word[i];
  }
  // version
  output->version = input->version;
  // total_packet_len
  output->total_packet_len = input->total_packet_len;
  // platform
  output->platform = input->platform;
  // frame_number
  output->frame_number = input->frame_number;
  // time_cpu_cycles
  output->time_cpu_cycles = input->time_cpu_cycles;
  // num_detected_obj
  output->num_detected_obj = input->num_detected_obj;
  // num_tlvs
  output->num_tlvs = input->num_tlvs;
  // subframe_number
  output->subframe_number = input->subframe_number;
  return true;
}

mmwave_radar_interfaces__msg__Header *
mmwave_radar_interfaces__msg__Header__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mmwave_radar_interfaces__msg__Header * msg = (mmwave_radar_interfaces__msg__Header *)allocator.allocate(sizeof(mmwave_radar_interfaces__msg__Header), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(mmwave_radar_interfaces__msg__Header));
  bool success = mmwave_radar_interfaces__msg__Header__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
mmwave_radar_interfaces__msg__Header__destroy(mmwave_radar_interfaces__msg__Header * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    mmwave_radar_interfaces__msg__Header__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
mmwave_radar_interfaces__msg__Header__Sequence__init(mmwave_radar_interfaces__msg__Header__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mmwave_radar_interfaces__msg__Header * data = NULL;

  if (size) {
    data = (mmwave_radar_interfaces__msg__Header *)allocator.zero_allocate(size, sizeof(mmwave_radar_interfaces__msg__Header), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = mmwave_radar_interfaces__msg__Header__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        mmwave_radar_interfaces__msg__Header__fini(&data[i - 1]);
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
mmwave_radar_interfaces__msg__Header__Sequence__fini(mmwave_radar_interfaces__msg__Header__Sequence * array)
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
      mmwave_radar_interfaces__msg__Header__fini(&array->data[i]);
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

mmwave_radar_interfaces__msg__Header__Sequence *
mmwave_radar_interfaces__msg__Header__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  mmwave_radar_interfaces__msg__Header__Sequence * array = (mmwave_radar_interfaces__msg__Header__Sequence *)allocator.allocate(sizeof(mmwave_radar_interfaces__msg__Header__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = mmwave_radar_interfaces__msg__Header__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
mmwave_radar_interfaces__msg__Header__Sequence__destroy(mmwave_radar_interfaces__msg__Header__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    mmwave_radar_interfaces__msg__Header__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
mmwave_radar_interfaces__msg__Header__Sequence__are_equal(const mmwave_radar_interfaces__msg__Header__Sequence * lhs, const mmwave_radar_interfaces__msg__Header__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!mmwave_radar_interfaces__msg__Header__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
mmwave_radar_interfaces__msg__Header__Sequence__copy(
  const mmwave_radar_interfaces__msg__Header__Sequence * input,
  mmwave_radar_interfaces__msg__Header__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(mmwave_radar_interfaces__msg__Header);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    mmwave_radar_interfaces__msg__Header * data =
      (mmwave_radar_interfaces__msg__Header *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!mmwave_radar_interfaces__msg__Header__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          mmwave_radar_interfaces__msg__Header__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!mmwave_radar_interfaces__msg__Header__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
