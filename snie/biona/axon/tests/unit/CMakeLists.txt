cmake_minimum_required(VERSION 3.25)

# Unit test targets

add_executable(test_safe_log test_safe_log.cpp)
target_link_libraries(test_safe_log PRIVATE biona_security biona_core)
add_test(NAME test_safe_log COMMAND test_safe_log)

add_executable(test_secret_manager test_secret_manager.cpp)
target_link_libraries(test_secret_manager PRIVATE biona_security biona_core)
add_test(NAME test_secret_manager COMMAND test_secret_manager)

add_executable(test_vad test_vad.cpp)
target_link_libraries(test_vad PRIVATE biona_core)
add_test(NAME test_vad COMMAND test_vad)
add_executable(test_signal_schema test_signal_schema.cpp)
target_link_libraries(test_signal_schema PRIVATE biona_core)
add_test(NAME test_signal_schema COMMAND test_signal_schema)

add_executable(test_spsc_queue test_spsc_queue.cpp)
target_link_libraries(test_spsc_queue PRIVATE biona_core)
add_test(NAME test_spsc_queue COMMAND test_spsc_queue)
