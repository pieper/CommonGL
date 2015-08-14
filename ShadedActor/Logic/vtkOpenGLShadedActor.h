/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkOpenGLShadedActor.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkOpenGLShadedActor - OpenGL actor
// .SECTION Description
// vtkOpenGLShadedActor is a concrete implementation of the abstract class vtkActor.
// vtkOpenGLShadedActor interfaces to the OpenGL rendering library.
// vtkOpenGLShadedActor allows the user to supply custom shaders to program the GPU

#ifndef __vtkOpenGLShadedActor_h
#define __vtkOpenGLShadedActor_h

// Slicer includes
#include "vtkSlicerModuleLogic.h"

#include "vtkActor.h"
#include "vtkImageData.h"

class vtkImageData;
class vtkOpenGLRenderer;

#include "vtkSlicerShadedActorModuleLogicExport.h"


class VTK_SLICER_SHADEDACTOR_MODULE_LOGIC_EXPORT vtkOpenGLShadedActor : public vtkActor
{
protected:
  
public:
  static vtkOpenGLShadedActor *New();
  vtkTypeMacro(vtkOpenGLShadedActor,vtkActor);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Actual actor render method.
  void Render(vtkRenderer *ren, vtkMapper *mapper);

  // Description:
  // Rebuild the shader program if needed
  bool UpdateProgram();

  // Description:
  // Reload the texture if needed
  bool UpdateTexture();

  // Description:
  // The strings defining the shaders
  vtkGetStringMacro(VertexShaderSource);
  vtkSetStringMacro(VertexShaderSource);
  vtkGetStringMacro(FragmentShaderSource);
  vtkSetStringMacro(FragmentShaderSource);

  // Description:
  // The 3D texture to use
  vtkGetObjectMacro(TextureImageData, vtkImageData);
  vtkSetObjectMacro(TextureImageData, vtkImageData);

  // Description:
  // Always assume the shaders will be translucent to some extent
  // so they will render after other geometry
  virtual int HasTranslucentPolygonalGeometry()
    { return 1; }

protected:
  vtkOpenGLShadedActor();
  ~vtkOpenGLShadedActor();

private:
  vtkOpenGLShadedActor(const vtkOpenGLShadedActor&);  // Not implemented.
  void operator=(const vtkOpenGLShadedActor&);  // Not implemented.

  char *VertexShaderSource;
  char *FragmentShaderSource;
  vtkImageData *TextureImageData;

  vtkTypeUInt32 ProgramObject; // same as GLuint: https://www.opengl.org/wiki/OpenGL_Type
  unsigned long ProgramObjectMTime;
  vtkTypeUInt32 TextureID;
  unsigned long TextureMTime;
};

#endif

