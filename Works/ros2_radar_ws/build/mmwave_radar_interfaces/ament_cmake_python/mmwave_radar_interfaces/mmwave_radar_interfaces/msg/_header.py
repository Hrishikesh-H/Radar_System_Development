# generated from rosidl_generator_py/resource/_idl.py.em
# with input from mmwave_radar_interfaces:msg/Header.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

# Member 'magic_word'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Header(type):
    """Metaclass of message 'Header'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('mmwave_radar_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'mmwave_radar_interfaces.msg.Header')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__header
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__header
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__header
            cls._TYPE_SUPPORT = module.type_support_msg__msg__header
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__header

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Header(metaclass=Metaclass_Header):
    """Message class 'Header'."""

    __slots__ = [
        '_magic_word',
        '_version',
        '_total_packet_len',
        '_platform',
        '_frame_number',
        '_time_cpu_cycles',
        '_num_detected_obj',
        '_num_tlvs',
        '_subframe_number',
    ]

    _fields_and_field_types = {
        'magic_word': 'uint8[8]',
        'version': 'uint32',
        'total_packet_len': 'uint32',
        'platform': 'uint32',
        'frame_number': 'uint32',
        'time_cpu_cycles': 'uint32',
        'num_detected_obj': 'uint32',
        'num_tlvs': 'uint32',
        'subframe_number': 'uint32',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 8),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        if 'magic_word' not in kwargs:
            self.magic_word = numpy.zeros(8, dtype=numpy.uint8)
        else:
            self.magic_word = numpy.array(kwargs.get('magic_word'), dtype=numpy.uint8)
            assert self.magic_word.shape == (8, )
        self.version = kwargs.get('version', int())
        self.total_packet_len = kwargs.get('total_packet_len', int())
        self.platform = kwargs.get('platform', int())
        self.frame_number = kwargs.get('frame_number', int())
        self.time_cpu_cycles = kwargs.get('time_cpu_cycles', int())
        self.num_detected_obj = kwargs.get('num_detected_obj', int())
        self.num_tlvs = kwargs.get('num_tlvs', int())
        self.subframe_number = kwargs.get('subframe_number', int())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if all(self.magic_word != other.magic_word):
            return False
        if self.version != other.version:
            return False
        if self.total_packet_len != other.total_packet_len:
            return False
        if self.platform != other.platform:
            return False
        if self.frame_number != other.frame_number:
            return False
        if self.time_cpu_cycles != other.time_cpu_cycles:
            return False
        if self.num_detected_obj != other.num_detected_obj:
            return False
        if self.num_tlvs != other.num_tlvs:
            return False
        if self.subframe_number != other.subframe_number:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def magic_word(self):
        """Message field 'magic_word'."""
        return self._magic_word

    @magic_word.setter
    def magic_word(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'magic_word' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 8, \
                "The 'magic_word' numpy.ndarray() must have a size of 8"
            self._magic_word = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 8 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'magic_word' field must be a set or sequence with length 8 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._magic_word = numpy.array(value, dtype=numpy.uint8)

    @builtins.property
    def version(self):
        """Message field 'version'."""
        return self._version

    @version.setter
    def version(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'version' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'version' field must be an unsigned integer in [0, 4294967295]"
        self._version = value

    @builtins.property
    def total_packet_len(self):
        """Message field 'total_packet_len'."""
        return self._total_packet_len

    @total_packet_len.setter
    def total_packet_len(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'total_packet_len' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'total_packet_len' field must be an unsigned integer in [0, 4294967295]"
        self._total_packet_len = value

    @builtins.property
    def platform(self):
        """Message field 'platform'."""
        return self._platform

    @platform.setter
    def platform(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'platform' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'platform' field must be an unsigned integer in [0, 4294967295]"
        self._platform = value

    @builtins.property
    def frame_number(self):
        """Message field 'frame_number'."""
        return self._frame_number

    @frame_number.setter
    def frame_number(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'frame_number' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'frame_number' field must be an unsigned integer in [0, 4294967295]"
        self._frame_number = value

    @builtins.property
    def time_cpu_cycles(self):
        """Message field 'time_cpu_cycles'."""
        return self._time_cpu_cycles

    @time_cpu_cycles.setter
    def time_cpu_cycles(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'time_cpu_cycles' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'time_cpu_cycles' field must be an unsigned integer in [0, 4294967295]"
        self._time_cpu_cycles = value

    @builtins.property
    def num_detected_obj(self):
        """Message field 'num_detected_obj'."""
        return self._num_detected_obj

    @num_detected_obj.setter
    def num_detected_obj(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'num_detected_obj' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'num_detected_obj' field must be an unsigned integer in [0, 4294967295]"
        self._num_detected_obj = value

    @builtins.property
    def num_tlvs(self):
        """Message field 'num_tlvs'."""
        return self._num_tlvs

    @num_tlvs.setter
    def num_tlvs(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'num_tlvs' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'num_tlvs' field must be an unsigned integer in [0, 4294967295]"
        self._num_tlvs = value

    @builtins.property
    def subframe_number(self):
        """Message field 'subframe_number'."""
        return self._subframe_number

    @subframe_number.setter
    def subframe_number(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'subframe_number' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'subframe_number' field must be an unsigned integer in [0, 4294967295]"
        self._subframe_number = value
