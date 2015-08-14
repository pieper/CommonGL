cmake_minimum_required(VERSION 2.8.9)

project(CommonGL)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/CommonGL")
set(EXTENSION_CATEGORY "Rendering")
set(EXTENSION_CONTRIBUTORS "Steve Pieper (Isomics, Inc.)")
set(EXTENSION_DESCRIPTION "A new infrastructure for rendering.")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/CommonGL.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/CommonGL/Screenshots/1.png")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ShadedActor)
add_subdirectory(ShadedModels)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
