Warning - experimental code.

The idea of this repository is to house some experiments in GLSL programming for volume rendering and processing.

Right now it's formulated as a Slicer Extension consisting of some C++ code (vtkOpenGLShadedActor) that manages
the OpenGL API and some python code (ShadedModels) that sets up the VTK context and defines the shaders.

In the ideal world the GLSL shader code should be independent of anything related to VTK or Slicer so
that it can be re-used in a variety of contexts, specifically in WebGL and OpenGL ES applications.

The exact form this will eventually take is still the subject of active investigation.
Initial efforts are focused on volume rendering.



Here's an example of volume rendering in a fragment shader using volume texture data from Slicer: 
![Volume rendering GLSL fragment shader in Slicer](https://raw.githubusercontent.com/pieper/CommonGL/master/images/glsl-brain2.png "Volume rendering GLSL fragment shader in Slicer")
