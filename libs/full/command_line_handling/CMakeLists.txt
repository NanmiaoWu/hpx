# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(command_line_handling_headers
    hpx/command_line_handling/command_line_handling.hpp
    hpx/command_line_handling/late_command_line_handling.hpp
    hpx/command_line_handling/parse_command_line.hpp
)

# cmake-format: off
set(command_line_handling_compat_headers
    hpx/util/command_line_handling.hpp => hpx/modules/command_line_handling.hpp
    hpx/util/parse_command_line.hpp => hpx/modules/command_line_handling.hpp
)
# cmake-format: on

set(command_line_handling_sources
    command_line_handling.cpp late_command_line_handling.cpp
    parse_command_line.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full command_line_handling
  GLOBAL_HEADER_GEN ON
  SOURCES ${command_line_handling_sources}
  HEADERS ${command_line_handling_headers}
  COMPAT_HEADERS ${command_line_handling_compat_headers}
  DEPENDENCIES hpx_core
  CMAKE_SUBDIRS tests
)
