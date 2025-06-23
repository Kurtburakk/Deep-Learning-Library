# Your existing working code:
find_package(GTest REQUIRED CONFIG)
include(GoogleTest)
add_executable(${PROJECT_NAME}Tests criterions.cpp functions.cpp operations.cpp optimizers.cpp test_conv2d.cpp)
target_link_libraries(${PROJECT_NAME}Tests PRIVATE ${PROJECT_NAME} GTest::gtest GTest::gtest_main GTest::gmock)
target_compile_features(${PROJECT_NAME}Tests PRIVATE cxx_std_17)
gtest_add_tests(TARGET ${PROJECT_NAME}Tests)

# Your broken code (FIXED):
add_executable(our_test test.cpp)
target_link_libraries(our_test PRIVATE cabernet)
target_include_directories(our_test PRIVATE ../include)
target_compile_features(our_test PRIVATE cxx_std_17)
