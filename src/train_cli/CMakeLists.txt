# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# lfs_train - headless trainer of the LFS_TRAINER_ONLY build

include(GNUInstallDirs)

add_executable(lfs_train main.cpp)

add_dependencies(lfs_train lfs_core_abi_stamp lfs_git_version)

target_include_directories(lfs_train BEFORE PRIVATE
    "${LFS_CORE_ABI_INCLUDE_DIR}"
)
target_include_directories(lfs_train PRIVATE
    ${CMAKE_SOURCE_DIR}/src/app/include  # headless_run_coordinator.hpp
    ${CMAKE_SOURCE_DIR}/src              # training/, core/event_bridge/
    ${CMAKE_BINARY_DIR}/include          # git_version.h
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(lfs_train PRIVATE
    lfs_training
    lfs_io
    lfs_core
    CUDA::cudart
    spdlog::spdlog
)

target_compile_definitions(lfs_train PRIVATE LFS_MIN_SM=${LFS_RUNTIME_MIN_SM})
target_compile_features(lfs_train PRIVATE cxx_std_23)
set_target_properties(lfs_train PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
lfs_add_windows_version_resource(lfs_train "LichtFeld Studio trainer")
lfs_add_windows_version_resource(lfs_core "LichtFeld Studio core and tensor runtime")
lfs_add_windows_version_resource(lfs_diagnostics "LichtFeld Studio diagnostics")
lfs_add_windows_version_resource(lfs_logger "LichtFeld Studio logging")
lfs_add_windows_version_resource(lfs_event_bridge "LichtFeld Studio event bridge")

get_property(_lfs_train_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(WIN32 AND _lfs_train_multi_config)
    set_target_properties(lfs_train PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>")
endif()

if(UNIX)
    set(_lfs_train_rpath "$ORIGIN" "$ORIGIN/lib" "${CUDAToolkit_LIBRARY_DIR}")
    if(DEFINED VCPKG_TARGET_TRIPLET)
        list(APPEND _lfs_train_rpath "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib")
    endif()
    if(TARGET OpenMeshCore)
        list(APPEND _lfs_train_rpath "$<TARGET_FILE_DIR:OpenMeshCore>")
    endif()
    list(JOIN _lfs_train_rpath ":" _lfs_train_rpath)
    set_target_properties(lfs_train PROPERTIES
        BUILD_RPATH "${_lfs_train_rpath}"
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
    )
endif()

# nvImageCodec runtime and its JPEG extension (GPU JPEG decode) next to the executable.
if(TARGET nvimgcodec)
    add_dependencies(lfs_train nvimgcodec)
    add_custom_command(TARGET lfs_train POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:nvimgcodec>" "$<TARGET_FILE_DIR:lfs_train>"
        COMMENT "Copying nvImageCodec library for lfs_train"
    )
    if(UNIX)
        add_custom_command(TARGET lfs_train POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "$<TARGET_FILE_NAME:nvimgcodec>" "$<TARGET_FILE_DIR:lfs_train>/libnvimgcodec.so.0"
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "libnvimgcodec.so.0" "$<TARGET_FILE_DIR:lfs_train>/libnvimgcodec.so"
        )
    endif()
endif()
if(TARGET nvjpeg_ext)
    add_dependencies(lfs_train nvjpeg_ext)
    add_custom_command(TARGET lfs_train POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:lfs_train>/extensions"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:nvjpeg_ext>" "$<TARGET_FILE_DIR:lfs_train>/extensions/"
        COMMENT "Copying nvjpeg extension for lfs_train"
    )
endif()

# JPEG 2000 extension and its runtime: the depth/normal sidecar cache and 16-bit image cache encode JPEG 2000.
if(TARGET nvjpeg2k_ext)
    add_dependencies(lfs_train nvjpeg2k_ext)
    add_custom_command(TARGET lfs_train POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:lfs_train>/extensions"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:nvjpeg2k_ext>" "$<TARGET_FILE_DIR:lfs_train>/extensions/"
        COMMENT "Copying nvjpeg2k extension for lfs_train"
    )

    if(WITH_DYNAMIC_NVJPEG2K)
        if(CUDAToolkit_VERSION_MAJOR)
            set(_lfs_train_nvjpeg2k_cuda_major "${CUDAToolkit_VERSION_MAJOR}")
        else()
            set(_lfs_train_nvjpeg2k_cuda_major "12")
        endif()
        set(_lfs_train_nvjpeg2k_runtime_dir
            "${CMAKE_BINARY_DIR}/_deps/nvjpeg2k_headers-src/lib/${_lfs_train_nvjpeg2k_cuda_major}")
        if(WIN32)
            set(_lfs_train_nvjpeg2k_runtime_file "${_lfs_train_nvjpeg2k_runtime_dir}/nvjpeg2k_0.dll")
            if(NOT EXISTS "${_lfs_train_nvjpeg2k_runtime_file}")
                unset(_lfs_train_nvjpeg2k_runtime_file)
            endif()
        elseif(UNIX)
            file(GLOB _lfs_train_nvjpeg2k_runtime_candidates
                "${_lfs_train_nvjpeg2k_runtime_dir}/libnvjpeg2k.so.*.*")
            list(SORT _lfs_train_nvjpeg2k_runtime_candidates)
            if(_lfs_train_nvjpeg2k_runtime_candidates)
                list(GET _lfs_train_nvjpeg2k_runtime_candidates -1 _lfs_train_nvjpeg2k_runtime_file)
            endif()
        endif()

        if(DEFINED _lfs_train_nvjpeg2k_runtime_file)
            get_filename_component(_lfs_train_nvjpeg2k_runtime_name
                "${_lfs_train_nvjpeg2k_runtime_file}" NAME)
            add_custom_command(TARGET lfs_train POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_lfs_train_nvjpeg2k_runtime_file}"
                    "$<TARGET_FILE_DIR:lfs_train>/${_lfs_train_nvjpeg2k_runtime_name}"
                COMMENT "Copying nvjpeg2k runtime library for lfs_train"
            )
            if(UNIX)
                add_custom_command(TARGET lfs_train POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                        "${_lfs_train_nvjpeg2k_runtime_name}" "$<TARGET_FILE_DIR:lfs_train>/libnvjpeg2k.so.0"
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                        "libnvjpeg2k.so.0" "$<TARGET_FILE_DIR:lfs_train>/libnvjpeg2k.so"
                )
            endif()
        else()
            message(WARNING
                "WITH_DYNAMIC_NVJPEG2K is ON but no nvjpeg2k runtime was found in "
                "${_lfs_train_nvjpeg2k_runtime_dir}")
        endif()
    endif()
endif()

# Windows: CUDA runtime and vcpkg DLLs next to the executable.
if(WIN32)
    file(GLOB _lfs_train_cuda_dlls
        "${CUDAToolkit_BIN_DIR}/cudart64*.dll" "${CUDAToolkit_BIN_DIR}/curand64*.dll"
        "${CUDAToolkit_BIN_DIR}/x64/cudart64*.dll" "${CUDAToolkit_BIN_DIR}/x64/curand64*.dll")
    if(_lfs_train_cuda_dlls)
        add_custom_command(TARGET lfs_train POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_lfs_train_cuda_dlls} "$<TARGET_FILE_DIR:lfs_train>"
            COMMENT "Copying CUDA runtime DLLs for lfs_train"
        )
    endif()
    if(DEFINED VCPKG_TARGET_TRIPLET)
        set(_lfs_train_vcpkg_bin "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}")
        file(GLOB _lfs_train_release_dlls CONFIGURE_DEPENDS "${_lfs_train_vcpkg_bin}/bin/*.dll")
        file(GLOB _lfs_train_debug_dlls CONFIGURE_DEPENDS "${_lfs_train_vcpkg_bin}/debug/bin/*.dll")
        if(_lfs_train_release_dlls)
            add_custom_command(TARGET lfs_train POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_lfs_train_release_dlls} "$<TARGET_FILE_DIR:lfs_train>"
                COMMENT "Copying vcpkg release DLLs for lfs_train"
            )
        endif()
        if(_lfs_train_debug_dlls)
            add_custom_command(TARGET lfs_train POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
                    ${_lfs_train_debug_dlls} "$<TARGET_FILE_DIR:lfs_train>"
                COMMENT "Copying vcpkg debug DLLs for lfs_train (Debug only)"
            )
        endif()
    endif()
endif()

install(TARGETS lfs_train RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS lfs_core lfs_diagnostics lfs_logger lfs_event_bridge
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(TARGET nvimgcodec)
    install(TARGETS nvimgcodec
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
if(TARGET nvjpeg_ext)
    # getExtensionsDir() looks for extensions/ next to bin/.
    install(TARGETS nvjpeg_ext
        LIBRARY DESTINATION extensions
        RUNTIME DESTINATION extensions
    )
endif()
if(TARGET nvjpeg2k_ext)
    install(TARGETS nvjpeg2k_ext
        LIBRARY DESTINATION extensions
        RUNTIME DESTINATION extensions
    )
    if(DEFINED _lfs_train_nvjpeg2k_runtime_file)
        if(WIN32)
            install(FILES "${_lfs_train_nvjpeg2k_runtime_file}" DESTINATION ${CMAKE_INSTALL_BINDIR})
        else()
            install(FILES "${_lfs_train_nvjpeg2k_runtime_file}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
            install(CODE "
                execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
                    \"${_lfs_train_nvjpeg2k_runtime_name}\"
                    \"libnvjpeg2k.so.0\"
                    WORKING_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
                execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
                    \"libnvjpeg2k.so.0\"
                    \"libnvjpeg2k.so\"
                    WORKING_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
            ")
        endif()
    endif()
endif()

message(STATUS "Trainer-only build: lfs_train (no GUI, Python, video, MCP/TCP, MoGe)")
