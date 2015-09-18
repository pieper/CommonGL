/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkOpenGLTextureImage.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkOpenGLTextureImage.h"

#include "vtkDataArray.h"
#include "vtkImageData.h"
#include "vtkMapper.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include "vtkOpenGLError.h"
#include "vtkOpenGLRenderer.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"

#include "vtkOpenGL.h"
#include <math.h>

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkOpenGLTextureImage);

//----------------------------------------------------------------------------
vtkOpenGLTextureImage::vtkOpenGLTextureImage()
{
  this->ShaderComputation = NULL;
  this->ImageData = NULL;
  this->TextureName = 0;
  this->TextureMTime = 0;
}

//----------------------------------------------------------------------------
vtkOpenGLTextureImage::~vtkOpenGLTextureImage()
{
  this->SetShaderComputation(NULL);
  this->SetImageData(NULL);
}

//----------------------------------------------------------------------------
// adapted from Rendering/OpenGL2/vtkTextureObject.cxx
static GLenum vtkScalarTypeToGLType(int vtk_scalar_type)
{
  // DON'T DEAL with VTK_CHAR as this is platform dependent.
  switch (vtk_scalar_type)
    {
  case VTK_SIGNED_CHAR:
    return GL_BYTE;

  case VTK_UNSIGNED_CHAR:
    return GL_UNSIGNED_BYTE;

  case VTK_SHORT:
    return GL_SHORT;

  case VTK_UNSIGNED_SHORT:
    return GL_UNSIGNED_SHORT;

  case VTK_INT:
    return GL_INT;

  case VTK_UNSIGNED_INT:
    return GL_UNSIGNED_INT;

  case VTK_FLOAT:
  case VTK_VOID: // used for depth component textures.
    return GL_FLOAT;
    }
  return 0;
}

//----------------------------------------------------------------------------
// Reload the texture if needed
//
bool vtkOpenGLTextureImage::UpdateTexture()
{
  if (this->ImageData->GetMTime() > this->TextureMTime)
    {
    if (this->TextureName != 0)
      {
      glDeleteTextures (1, &(this->TextureName) );
      }
    this->TextureMTime = 0;
    }
  else
    {
    return true;
    }

  if ( this->ImageData == 0 )
    {
    return false;
    }
  if ( this->ImageData->GetNumberOfScalarComponents() != 1 )
    {
    vtkErrorMacro("Must have 1 component image data for texture");
    return false;
    }

  int dimensions[3];
  this->ImageData->GetDimensions(dimensions);
  vtkPointData *pointData = this->ImageData->GetPointData();
  vtkDataArray *scalars = pointData->GetScalars();
  void *pixels = scalars->GetVoidPointer(0);

  vtkOpenGLCheckErrorMacro("before uploading");

  glGenTextures(1, &(this->TextureName));
  glBindTexture(GL_TEXTURE_3D, this->TextureName);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexImage3D(/* target */            GL_TEXTURE_3D,
               /* level */             0,
               /* internal format */   1,
               /* width */             dimensions[0],
               /* height */            dimensions[1],
               /* depth */             dimensions[2],
               /* border */            0,
               /* format */            GL_LUMINANCE,
               /* type */              vtkScalarTypeToGLType(this->ImageData->GetScalarType()),
               /* pixels */            pixels
  );
  vtkOpenGLCheckErrorMacro("after uploading");

  this->TextureMTime = this->GetMTime();
  return true;
}

void vtkOpenGLTextureImage::Activate(vtkTypeUInt32 unit)
{

  vtkOpenGLCheckErrorMacro("before activating");

  if (!this->ShaderComputation || !this->ShaderComputation->GetInitialized())
    {
    vtkErrorMacro("No initialized ShaderComputation instance is set.");
    return;
    }
  this->ShaderComputation->GetRenderWindow()->MakeCurrent();

  // TODO: check the actual number (also expose way to check the
  // number from a wrapped language).  For now use the minimum max value.
  // of the enums, which only go to 15 even though 48 are meant to be 
  // supported according to the OpenGL spec.
  // glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &texture_units)
  #define __TEXTURE_UNIT_COUNT 16
  if (unit > __TEXTURE_UNIT_COUNT-1)
    {
    vtkErrorMacro("Only " << __TEXTURE_UNIT_COUNT << " texture units are available.");
    return;
    }

  if (!this->UpdateTexture())
    {
    vtkErrorMacro("Could not update texture.");
    return;
    }

  // TODO: 
  glActiveTexture(GL_TEXTURE0 + unit);
  glBindTexture(GL_TEXTURE_3D, this->TextureName);

  vtkOpenGLCheckErrorMacro("after activating");
}

//----------------------------------------------------------------------------
void vtkOpenGLTextureImage::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  if ( this->ShaderComputation )
    {
    os << indent << "ShaderComputation: " << this->ShaderComputation << "\n";
    }
  else
    {
    os << indent << "ShaderComputation: (none)\n";
    }
  if ( this->ImageData )
    {
    os << indent << "ImageData: " << this->ImageData << "\n";
    }
  else
    {
    os << indent << "ImageData: (none)\n";
    }
  os << indent << "TextureName: " << this->TextureName << "\n";
  os << indent << "TextureMTime: " << this->TextureMTime << "\n";
}
