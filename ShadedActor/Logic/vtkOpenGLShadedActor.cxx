/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkOpenGLShadedActor.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkOpenGLShadedActor.h"

#include "vtkDataArray.h"
#include "vtkImageData.h"
#include "vtkMapper.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include "vtkOpenGLRenderer.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"

#include "vtkOpenGL.h"
#include <math.h>

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkOpenGLShadedActor);

//----------------------------------------------------------------------------
vtkOpenGLShadedActor::vtkOpenGLShadedActor()
{
  this->VertexShaderSource = NULL;
  this->FragmentShaderSource = NULL;
  this->TextureImageData = NULL;
  this->ProgramObject = 0;
  this->ProgramObjectMTime = 0;
  this->TextureID = 0;
  this->TextureMTime = 0;
}

//----------------------------------------------------------------------------
vtkOpenGLShadedActor::~vtkOpenGLShadedActor()
{
  this->SetVertexShaderSource(NULL);
  this->SetFragmentShaderSource(NULL);
  this->SetTextureImageData(NULL);
  if (this->ProgramObject > 0)
    {
    glDeleteProgram ( this->ProgramObject );
    this->ProgramObject = 0;
    }
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
///
// Create a shader object, load the shader source, and
// compile the shader.
//
static GLuint CompileShader ( vtkOpenGLShadedActor *self, GLenum type, const char *shaderSource )
{
  GLuint shader;
  GLint compiled;

  // Create the shader object
  shader = glCreateShader ( type );

  if ( shader == 0 )
    {
    return 0;
    }

  // Load the shader source
  glShaderSource ( shader, 1, &shaderSource, NULL );

  // Compile the shader
  glCompileShader ( shader );

  // Check the compile status
  glGetShaderiv ( shader, GL_COMPILE_STATUS, &compiled );

  if ( !compiled )
    {
    GLint infoLen = 0;

    glGetShaderiv ( shader, GL_INFO_LOG_LENGTH, &infoLen );

    if ( infoLen > 1 )
      {
      char *infoLog = (char *) malloc ( sizeof ( char ) * infoLen );

      glGetShaderInfoLog ( shader, infoLen, NULL, infoLog );
      vtkErrorWithObjectMacro (self, "Error compiling shader\n" << infoLog );

      free ( infoLog );
      }

      glDeleteShader ( shader );
      return 0;
   }
   return shader;
}

//----------------------------------------------------------------------------
// Rebuild the shader program if needed
//
bool vtkOpenGLShadedActor::UpdateProgram()
{
  GLuint vertexShader;
  GLuint fragmentShader;
  GLint linked;

  if (this->GetMTime() > this->ProgramObjectMTime)
    {
    if (this->ProgramObject != 0)
      {
      glDeleteProgram ( this->ProgramObject );
      }
    this->ProgramObjectMTime = 0;
    }
  else
    {
    return true;
    }

  // Load the vertex/fragment shaders
  vertexShader = CompileShader ( this, GL_VERTEX_SHADER, this->VertexShaderSource );
  fragmentShader = CompileShader ( this, GL_FRAGMENT_SHADER, this->FragmentShaderSource );

  // Create the program object
  this->ProgramObject = glCreateProgram ( );

  if ( this->ProgramObject == 0 )
    {
    return false;
    }

  glAttachShader ( this->ProgramObject, vertexShader );
  glAttachShader ( this->ProgramObject, fragmentShader );

  glLinkProgram ( this->ProgramObject );

  // Check the link status
  glGetProgramiv ( this->ProgramObject, GL_LINK_STATUS, &linked );

  if ( !linked )
    {
    // something went wrong, so emit error message if possible
    GLint infoLen = 0;
    glGetProgramiv ( this->ProgramObject, GL_INFO_LOG_LENGTH, &infoLen );

    if ( infoLen > 1 )
      {
      char *infoLog = (char *) malloc ( sizeof ( char ) * infoLen );

      glGetProgramInfoLog ( this->ProgramObject, infoLen, NULL, infoLog );
      vtkErrorMacro ( "Error linking program\n" << infoLog );

      free ( infoLog );
      }

    glDeleteProgram ( this->ProgramObject );
    return false;
    }

  this->ProgramObjectMTime = this->GetMTime();
  return true;
}

//----------------------------------------------------------------------------
// Reload the texture if needed
//
bool vtkOpenGLShadedActor::UpdateTexture()
{
  if (this->GetMTime() > this->TextureMTime)
    {
    if (this->TextureID != 0)
      {
      glDeleteTextures (1, &(this->TextureID) );
      }
    this->TextureMTime = 0;
    }
  else
    {
    return true;
    }

  if ( this->TextureImageData == 0 )
    {
    return false;
    }
  if ( this->TextureImageData->GetNumberOfScalarComponents() != 1 )
    {
    vtkErrorMacro("Must have 1 component image data for texture");
    return false;
    }

  int dimensions[3];
  this->TextureImageData->GetDimensions(dimensions);
  vtkPointData *pointData = this->TextureImageData->GetPointData();
  vtkDataArray *scalars = pointData->GetScalars();
  void *pixels = scalars->GetVoidPointer(0);
  short *shortPixels = (short *)pixels;

  glEnable(GL_TEXTURE_3D);
  glGenTextures(1, &(this->TextureID));
  glBindTexture(GL_TEXTURE_3D, this->TextureID);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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
               /* type */              vtkScalarTypeToGLType(this->TextureImageData->GetScalarType()),
               /* pixels */            pixels
  );

  this->TextureMTime = this->GetMTime();
  return true;
}

//----------------------------------------------------------------------------
// Actual scripted actor render method.
//
// This is essenitally the core of vtkOpenGLActor, but with option to specify
// the shader program source.
// It is not used directly, but the mapper available to access rendering parameters.
//
//
void vtkOpenGLShadedActor::Render(vtkRenderer *ren, vtkMapper *mapper)
{

  // bail out early if we aren't configured corretly
  if (this->VertexShaderSource == NULL || this->FragmentShaderSource == NULL)
    {
    vtkErrorMacro("Both vertex and fragment shaders are needed for a shaded actor.");
    return;
    }

  vtkPolyDataMapper *polyDataMapper = vtkPolyDataMapper::SafeDownCast(mapper);
  if (polyDataMapper == NULL)
    {
    vtkErrorMacro("Need a vtkPolyDataMapper.");
    return;
    }

  if (!this->UpdateProgram())
    {
    vtkErrorMacro("Could not update shader program.");
    return;
    }

  if (!this->UpdateTexture())
    {
    vtkErrorMacro("Could not update texture.");
    return;
    }

  GLfloat planeVertices[] = { -1.0f, -1.0f, 0.0f,
                              -1.0f,  1.0f, 0.0f,
                               1.0f,  1.0f, 0.0f,
                               1.0f, -1.0f, 0.0f
                        };
  GLfloat planeTextureCoordinates[] = { 0.0f, 1.0f,
                                        0.0f, 0.0f,
                                        1.0f, 0.0f,
                                        1.0f, 1.0f
                        };

  // Use the program object
  glUseProgram ( this->ProgramObject );

  // Put the vertices into a buffer and specify that their location 
  // is 0, meaning it will correspond to the first attribute of
  // the vertex shader.
  // Some help here:
  // http://www.morethantechnical.com/2013/11/09/vertex-array-objects-with-shaders-on-opengl-2-1-glsl-1-2-wcode/
  GLuint vertexLocation = glGetAttribLocation(this->ProgramObject, "vertexAttribute");
  GLuint planeVerticesBuffer;
  glGenBuffers(1, &planeVerticesBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, planeVerticesBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*3*4, planeVertices, GL_STATIC_DRAW);
  glEnableVertexAttribArray ( vertexLocation );
  glVertexAttribPointer ( vertexLocation, 3, GL_FLOAT, GL_FALSE, 0, 0 );

  // Put texture coordinates in a buffer
  GLuint textureCoordinatesLocation = glGetAttribLocation(this->ProgramObject, "textureCoordinateAttribute");
  GLuint textureCoordinatesBuffer;
  glGenBuffers(1, &textureCoordinatesBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, textureCoordinatesBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*2*4, planeTextureCoordinates, GL_STATIC_DRAW);
  glEnableVertexAttribArray ( textureCoordinatesLocation );
  glVertexAttribPointer ( textureCoordinatesLocation, 2, GL_FLOAT, GL_FALSE, 0, 0 );

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, this->TextureID);
  GLuint volumeSamplerLocation = glGetUniformLocation(this->ProgramObject, "volumeSampler");
  glUniform1i(volumeSamplerLocation, 0);

  glDrawArrays ( GL_QUADS, 0, 4 );

  // Don't use the program anymore
  glUseProgram ( 0 );


  // - add a vtkCollection of vtkImageData and make each one a texture3D (in order)
  // - bind the vertex array and the texture coordinates
  // - traverse the vtkPolyData from the mapper and draw the primitives (just triangle
  //   strips is good enough to start with

}

//----------------------------------------------------------------------------
void vtkOpenGLShadedActor::PrintSelf(ostream& os, vtkIndent indent)
{
  if ( this->VertexShaderSource )
    {
    os << indent << "VertexShaderSource: " << this->VertexShaderSource << "\n";
    }
  else
    {
    os << indent << "VertexShaderSource: (none)\n";
    }
  if ( this->FragmentShaderSource )
    {
    os << indent << "FragmentShaderSource: " << this->FragmentShaderSource << "\n";
    }
  else
    {
    os << indent << "FragmentShaderSource: (none)\n";
    }
  if ( this->TextureImageData )
    {
    os << indent << "TextureImageData: " << this->TextureImageData << "\n";
    }
  else
    {
    os << indent << "TextureImageData: (none)\n";
    }
  os << indent << "ProgramObject: " << this->ProgramObject << "\n";
  os << indent << "ProgramObjectMTime: " << this->ProgramObjectMTime << "\n";
  os << indent << "TextureID: " << this->TextureID << "\n";
  os << indent << "TextureMTime: " << this->TextureMTime << "\n";

  this->Superclass::PrintSelf(os,indent);
}
