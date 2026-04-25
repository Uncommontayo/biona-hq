# CMake generated Testfile for 
# Source directory: C:/Users/Home/Downloads/biona_axon/snie/biona/axon/tests/integration
# Build directory: C:/Users/Home/Downloads/biona_axon/snie/build_msvc/tests/integration
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[test_pipeline_mock]=] "C:/Users/Home/Downloads/biona_axon/snie/build_msvc/tests/integration/Debug/test_pipeline_mock.exe")
  set_tests_properties([=[test_pipeline_mock]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Home/Downloads/biona_axon/snie/biona/axon/tests/integration/CMakeLists.txt;5;add_test;C:/Users/Home/Downloads/biona_axon/snie/biona/axon/tests/integration/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[test_pipeline_mock]=] "C:/Users/Home/Downloads/biona_axon/snie/build_msvc/tests/integration/Release/test_pipeline_mock.exe")
  set_tests_properties([=[test_pipeline_mock]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Home/Downloads/biona_axon/snie/biona/axon/tests/integration/CMakeLists.txt;5;add_test;C:/Users/Home/Downloads/biona_axon/snie/biona/axon/tests/integration/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test([=[test_pipeline_mock]=] "C:/Users/Home/Downloads/biona_axon/snie/build_msvc/tests/integration/MinSizeRel/test_pipeline_mock.exe")
  set_tests_properties([=[test_pipeline_mock]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Home/Downloads/biona_axon/snie/biona/axon/tests/integration/CMakeLists.txt;5;add_test;C:/Users/Home/Downloads/biona_axon/snie/biona/axon/tests/integration/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test([=[test_pipeline_mock]=] "C:/Users/Home/Downloads/biona_axon/snie/build_msvc/tests/integration/RelWithDebInfo/test_pipeline_mock.exe")
  set_tests_properties([=[test_pipeline_mock]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Home/Downloads/biona_axon/snie/biona/axon/tests/integration/CMakeLists.txt;5;add_test;C:/Users/Home/Downloads/biona_axon/snie/biona/axon/tests/integration/CMakeLists.txt;0;")
else()
  add_test([=[test_pipeline_mock]=] NOT_AVAILABLE)
endif()
