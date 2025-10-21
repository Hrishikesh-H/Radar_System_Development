// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from mmwave_radar_interfaces:msg/Header.idl
// generated code does not contain a copyright notice
#include "mmwave_radar_interfaces/msg/detail/header__rosidl_typesupport_fastrtps_cpp.hpp"
#include "mmwave_radar_interfaces/msg/detail/header__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace mmwave_radar_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mmwave_radar_interfaces
cdr_serialize(
  const mmwave_radar_interfaces::msg::Header & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: magic_word
  {
    cdr << ros_message.magic_word;
  }
  // Member: version
  cdr << ros_message.version;
  // Member: total_packet_len
  cdr << ros_message.total_packet_len;
  // Member: platform
  cdr << ros_message.platform;
  // Member: frame_number
  cdr << ros_message.frame_number;
  // Member: time_cpu_cycles
  cdr << ros_message.time_cpu_cycles;
  // Member: num_detected_obj
  cdr << ros_message.num_detected_obj;
  // Member: num_tlvs
  cdr << ros_message.num_tlvs;
  // Member: subframe_number
  cdr << ros_message.subframe_number;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mmwave_radar_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  mmwave_radar_interfaces::msg::Header & ros_message)
{
  // Member: magic_word
  {
    cdr >> ros_message.magic_word;
  }

  // Member: version
  cdr >> ros_message.version;

  // Member: total_packet_len
  cdr >> ros_message.total_packet_len;

  // Member: platform
  cdr >> ros_message.platform;

  // Member: frame_number
  cdr >> ros_message.frame_number;

  // Member: time_cpu_cycles
  cdr >> ros_message.time_cpu_cycles;

  // Member: num_detected_obj
  cdr >> ros_message.num_detected_obj;

  // Member: num_tlvs
  cdr >> ros_message.num_tlvs;

  // Member: subframe_number
  cdr >> ros_message.subframe_number;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mmwave_radar_interfaces
get_serialized_size(
  const mmwave_radar_interfaces::msg::Header & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: magic_word
  {
    size_t array_size = 8;
    size_t item_size = sizeof(ros_message.magic_word[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: version
  {
    size_t item_size = sizeof(ros_message.version);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: total_packet_len
  {
    size_t item_size = sizeof(ros_message.total_packet_len);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: platform
  {
    size_t item_size = sizeof(ros_message.platform);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: frame_number
  {
    size_t item_size = sizeof(ros_message.frame_number);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: time_cpu_cycles
  {
    size_t item_size = sizeof(ros_message.time_cpu_cycles);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: num_detected_obj
  {
    size_t item_size = sizeof(ros_message.num_detected_obj);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: num_tlvs
  {
    size_t item_size = sizeof(ros_message.num_tlvs);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: subframe_number
  {
    size_t item_size = sizeof(ros_message.subframe_number);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_mmwave_radar_interfaces
max_serialized_size_Header(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;


  // Member: magic_word
  {
    size_t array_size = 8;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: version
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: total_packet_len
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: platform
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: frame_number
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: time_cpu_cycles
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: num_detected_obj
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: num_tlvs
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: subframe_number
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = mmwave_radar_interfaces::msg::Header;
    is_plain =
      (
      offsetof(DataType, subframe_number) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _Header__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const mmwave_radar_interfaces::msg::Header *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _Header__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<mmwave_radar_interfaces::msg::Header *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _Header__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const mmwave_radar_interfaces::msg::Header *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _Header__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_Header(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _Header__callbacks = {
  "mmwave_radar_interfaces::msg",
  "Header",
  _Header__cdr_serialize,
  _Header__cdr_deserialize,
  _Header__get_serialized_size,
  _Header__max_serialized_size
};

static rosidl_message_type_support_t _Header__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_Header__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace mmwave_radar_interfaces

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_mmwave_radar_interfaces
const rosidl_message_type_support_t *
get_message_type_support_handle<mmwave_radar_interfaces::msg::Header>()
{
  return &mmwave_radar_interfaces::msg::typesupport_fastrtps_cpp::_Header__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, mmwave_radar_interfaces, msg, Header)() {
  return &mmwave_radar_interfaces::msg::typesupport_fastrtps_cpp::_Header__handle;
}

#ifdef __cplusplus
}
#endif
