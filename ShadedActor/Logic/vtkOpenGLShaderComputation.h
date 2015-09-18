/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkOpenGLShaderComputation.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkOpenGLShaderComputation - OpenGL actor
// .SECTION Description
// vtkOpenGLShaderComputation is a way to perform GPU computations on vtk data.
// vtkOpenGLShaderComputation interfaces to the OpenGL rendering library.

#ifndef __vtkOpenGLShaderComputation_h
#define __vtkOpenGLShaderComputation_h

// Slicer includes
#include "vtkSlicerModuleLogic.h"

#include "vtkImageData.h"
#include "vtkRenderWindow.h"

class vtkImageData;

#include "vtkSlicerShadedActorModuleLogicExport.h"


class VTK_SLICER_SHADEDACTOR_MODULE_LOGIC_EXPORT vtkOpenGLShaderComputation : public vtkObject
{
protected:

public:
  static vtkOpenGLShaderComputation *New();
  vtkTypeMacro(vtkOpenGLShaderComputation,vtkObject);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Loads the required extensions
  void Initialize(vtkRenderWindow *renderWindow);

  // Description:
  // Rebuild the shader program if needed
  bool UpdateProgram();

  // Description:
  // Manage the OpenGL offscreen rendering framebuffer for computing
  bool AcquireFramebuffer();
  void ReleaseFramebuffer();

  // Description:
  // Perform the actual computation
  // Updates the texture and program if needed and then
  // renders a quadrilateral of to a renderbuffer the size
  // of the ResultImageData and uses the program
  // to perform the shading.
  void Compute();

  // Description:
  // The strings defining the shaders
  vtkGetStringMacro(VertexShaderSource);
  vtkSetStringMacro(VertexShaderSource);
  vtkGetStringMacro(FragmentShaderSource);
  vtkSetStringMacro(FragmentShaderSource);

  // Description:
  // The results of the computation.
  // Must be set with the desired dimensions before calling Compute.
  vtkGetObjectMacro(ResultImageData, vtkImageData);
  vtkSetObjectMacro(ResultImageData, vtkImageData);

  // Description:
  // Used internally to manage OpenGL context and extensions
  vtkGetObjectMacro(RenderWindow, vtkRenderWindow);
  vtkSetObjectMacro(RenderWindow, vtkRenderWindow);

  // Description:
  // Has the context been set up with a render window?
  vtkGetMacro(Initialized, bool);

protected:
  vtkOpenGLShaderComputation();
  ~vtkOpenGLShaderComputation();

private:
  vtkOpenGLShaderComputation(const vtkOpenGLShaderComputation&);  // Not implemented.
  void operator=(const vtkOpenGLShaderComputation&);  // Not implemented.

  bool Initialized;
  char *VertexShaderSource;
  char *FragmentShaderSource;
  vtkImageData *ResultImageData;

  vtkTypeUInt32 ProgramObject; // vtkTypeUInt32 same as GLuint: https://www.opengl.org/wiki/OpenGL_Type
  unsigned long ProgramObjectMTime;
  vtkTypeUInt32 FramebufferID;
  vtkTypeUInt32 ColorRenderbufferID;
  vtkTypeUInt32 DepthRenderbufferID;

  vtkRenderWindow *RenderWindow;
};

#endif

