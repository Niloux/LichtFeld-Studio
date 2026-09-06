# SPDX-License-Identifier: GPL-3.0-or-later
add_subdirectory(src/geometry)
add_subdirectory(src/diagnostics)
add_subdirectory(src/core)
add_subdirectory(src/io)
add_subdirectory(src/training)
add_executable(lfs-train src/train_app/main.cpp src/train_app/runtime.cpp)
target_link_libraries(lfs-train PRIVATE lfs_training lfs_io lfs_core CUDA::cudart)
target_include_directories(lfs-train PRIVATE
    "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/src/app/include" "${LFS_CORE_ABI_INCLUDE_DIR}" "${PROJECT_BINARY_DIR}/include")
target_compile_features(lfs-train PRIVATE cxx_std_23)
add_dependencies(lfs-train lfs_git_version lfs_core_abi_stamp)
set_target_properties(lfs-train PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")
if(NOT TARGET nvimgcodec)
    message(FATAL_ERROR "lfs-train requires nvImageCodec; initialize the repository submodules")
endif()
add_dependencies(lfs-train nvimgcodec)
add_custom_command(TARGET lfs-train POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:nvimgcodec>" "$<TARGET_FILE_DIR:lfs-train>"
    VERBATIM)
if(UNIX)
    add_custom_command(TARGET lfs-train POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "$<TARGET_FILE_NAME:nvimgcodec>" "$<TARGET_FILE_DIR:lfs-train>/$<TARGET_SONAME_FILE_NAME:nvimgcodec>"
        VERBATIM)
endif()
install(TARGETS nvimgcodec LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
foreach(codec IN ITEMS nvjpeg_ext nvjpeg2k_ext)
    if(TARGET ${codec})
        add_dependencies(lfs-train ${codec})
        add_custom_command(TARGET lfs-train POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:lfs-train>/extensions"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${codec}>" "$<TARGET_FILE_DIR:lfs-train>/extensions/"
            VERBATIM)
        install(TARGETS ${codec} LIBRARY DESTINATION lib/extensions RUNTIME DESTINATION extensions)
    endif()
endforeach()
# nvjpeg2k_ext loads the codec dynamically; it is not visible in target link dependencies.
if(NVJPEG2K_REDIST_ROOT AND UNIX)
    file(GLOB jpeg2k_runtime
        "${NVJPEG2K_REDIST_ROOT}/lib/libnvjpeg2k.so*"
        "${NVJPEG2K_REDIST_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}/libnvjpeg2k.so*")
    if(jpeg2k_runtime)
        add_custom_command(TARGET lfs-train POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${jpeg2k_runtime}
                "$<TARGET_FILE_DIR:lfs-train>/"
            VERBATIM)
        install(FILES ${jpeg2k_runtime} DESTINATION lib)
    endif()
endif()
# OpenMesh is a shared data-layer dependency on Unix in the vendored build.
# Give every shipped LFS library a relative runpath for a relocatable install.
foreach(runtime IN ITEMS lfs_core lfs_logger lfs_event_bridge lfs_diagnostics nvimgcodec OpenMeshCore)
    set_target_properties(${runtime} PROPERTIES INSTALL_RPATH "$ORIGIN")
endforeach()
install(TARGETS lfs-train lfs_core lfs_logger lfs_event_bridge lfs_diagnostics OpenMeshCore
    RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
foreach(forbidden IN ITEMS lfs_visualizer lfs_rendering lfs_sequencer lfs_mcp
        lfs_tcp lfs_video lfs_py lfs_python_utils lfs_preprocessing lfs_tree_sitter Zep
        spz_lib lfs_tinyusdz lfs_io_rad_quant_cuda Vulkan::Vulkan SDL3::SDL3 RmlUi::RmlUi)
    if(TARGET ${forbidden})
        message(FATAL_ERROR "train profile unexpectedly configured ${forbidden}")
    endif()
endforeach()

target_compile_definitions(lfs-train PRIVATE LFS_MIN_SM=${LFS_RUNTIME_MIN_SM})

option(LFS_BUILD_TRAIN_TESTS "Register train profile and CLI contract tests (no GUI dependencies)" OFF)
if(LFS_BUILD_TRAIN_TESTS)
    add_executable(lfs-config-test "${PROJECT_SOURCE_DIR}/tests/test_train_config.cpp")
    target_link_libraries(lfs-config-test PRIVATE lfs_core)
    add_executable(lfs-sky-test "${PROJECT_SOURCE_DIR}/tests/test_sky_background.cpp")
    target_include_directories(lfs-sky-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(lfs-sky-test PRIVATE lfs_training)
    add_executable(lfs-ppisp-test "${PROJECT_SOURCE_DIR}/tests/test_ppisp_color_domain.cpp")
    target_include_directories(lfs-ppisp-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(lfs-ppisp-test PRIVATE lfs_training)
    enable_testing()
    add_test(NAME train_config COMMAND lfs-config-test
        "${PROJECT_SOURCE_DIR}/configs/contextcapture_gaussian_sky.json")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME train_profile_contract
        COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/test_train_profile.py")
    add_test(NAME train_build_graph
        COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tools/check_train_build.py"
            "${CMAKE_BINARY_DIR}")
    add_test(NAME train_cli_version COMMAND lfs-train --version)
endif()
