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

class vtkImageData;
class vtkRenderer;

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
  void Initialize(vtkRenderer *renderer);

  // Description:
  // Rebuild the shader program if needed
  bool UpdateProgram();

  // Description:
  // Reload the texture if needed
  bool UpdateTexture();

  // Description:
  // Setup target for rendering result
  bool SetupFramebuffer();

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
  // The 3D texture to use as input.
  vtkGetObjectMacro(TextureImageData, vtkImageData);
  vtkSetObjectMacro(TextureImageData, vtkImageData);

  // Description:
  // The results of the computation.
  // Must be set with the desired dimensions before calling Compute.
  vtkGetObjectMacro(ResultImageData, vtkImageData);
  vtkSetObjectMacro(ResultImageData, vtkImageData);

protected:
  vtkOpenGLShaderComputation();
  ~vtkOpenGLShaderComputation();

private:
  vtkOpenGLShaderComputation(const vtkOpenGLShaderComputation&);  // Not implemented.
  void operator=(const vtkOpenGLShaderComputation&);  // Not implemented.

  bool Initialized;
  char *VertexShaderSource;
  char *FragmentShaderSource;
  vtkImageData *TextureImageData;
  vtkImageData *ResultImageData;

  vtkTypeUInt32 ProgramObject; // vtkTypeUInt32 same as GLuint: https://www.opengl.org/wiki/OpenGL_Type
  unsigned long ProgramObjectMTime;
  vtkTypeUInt32 TextureID;
  unsigned long TextureMTime;
  vtkTypeUInt32 FramebufferID;
  vtkTypeUInt32 ColorRenderbufferID;
  vtkTypeUInt32 DepthRenderbufferID;
};

#endif

