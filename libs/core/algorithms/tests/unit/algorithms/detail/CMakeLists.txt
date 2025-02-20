# Copyright (c) 2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests test_insertion_sort test_parallel_stable_sort test_sample_sort
          test_spin_sort
)

set(test_parallel_stable_sort_PARAMETERS THREADS_PER_LOCALITY 4)
set(test_sample_sort_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Core/Algorithms/Detail")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.algorithms.detail" ${test} ${${test}_PARAMETERS})
endforeach()
