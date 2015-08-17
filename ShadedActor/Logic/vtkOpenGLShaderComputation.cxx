/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkOpenGLShaderComputation.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkOpenGLShaderComputation.h"

#include "vtkDataArray.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkOpenGLError.h"
#include "vtkOpenGLExtensionManager.h"
#include "vtkOpenGLRenderWindow.h"
#include "vtkPointData.h"

#include "vtkOpenGL.h"
#include "vtkgl.h"

#include <math.h>

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkOpenGLShaderComputation);

//----------------------------------------------------------------------------
vtkOpenGLShaderComputation::vtkOpenGLShaderComputation()
{
  this->Initialized = false;
  this->VertexShaderSource = NULL;
  this->FragmentShaderSource = NULL;
  this->TextureImageData = NULL;
  this->ResultImageData = NULL;
  this->ProgramObject = 0;
  this->ProgramObjectMTime = 0;
  this->TextureID = 0;
  this->TextureMTime = 0;

  this->RenderWindow = vtkRenderWindow::New();
  this->RenderWindow->SetOffScreenRendering(1);
  this->Initialize(this->RenderWindow);
}

//----------------------------------------------------------------------------
vtkOpenGLShaderComputation::~vtkOpenGLShaderComputation()
{
  this->SetVertexShaderSource(NULL);
  this->SetFragmentShaderSource(NULL);
  this->SetTextureImageData(NULL);
  this->SetResultImageData(NULL);
  if (this->ProgramObject > 0)
    {
    glDeleteProgram ( this->ProgramObject );
    this->ProgramObject = 0;
    }
  if (this->TextureID > 0)
    {
    glDeleteTextures(1, &(this->TextureID));
    this->TextureID = 0;
    }
  this->SetRenderWindow(NULL);
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
static GLuint CompileShader ( vtkOpenGLShaderComputation *self, GLenum type, const char *shaderSource )
{
  vtkOpenGLClearErrorMacro();

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
      vtkOpenGLStaticCheckErrorMacro("after deleting bad shader");
      return 0;
    }
  vtkOpenGLStaticCheckErrorMacro("after compiling shader");
  return shader;
}

//----------------------------------------------------------------------------
// Rebuild the shader program if needed
//
bool vtkOpenGLShaderComputation::UpdateProgram()
{
  vtkOpenGLClearErrorMacro();
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

  if ( !vertexShader || !fragmentShader )
    {
    vtkOpenGLCheckErrorMacro("after failed compile");
    return false;
    }

  // Create the program object
  this->ProgramObject = glCreateProgram ( );

  if ( this->ProgramObject == 0 )
    {
    vtkOpenGLCheckErrorMacro("after failed program create");
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
    vtkOpenGLCheckErrorMacro("after failed program attachment");
    return false;
    }

  this->ProgramObjectMTime = this->GetMTime();
  vtkOpenGLCheckErrorMacro("after program creation");
  return true;
}

//----------------------------------------------------------------------------
// Reload the texture if needed
//
bool vtkOpenGLShaderComputation::UpdateTexture()
{
  vtkOpenGLClearErrorMacro();
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
  vtkOpenGLCheckErrorMacro("after texture update");
  return true;
}

//-----------------------------------------------------------------------------
void vtkOpenGLShaderComputation::Initialize(vtkRenderWindow *renderWindow)
{
  if (this->Initialized)
    {
    return;
    }

  vtkOpenGLRenderWindow *openGLRenderWindow = vtkOpenGLRenderWindow::SafeDownCast(renderWindow);
  if (!openGLRenderWindow)
    {
    vtkErrorMacro("Bad render window");
    return;
    }

  // load required extensions
  vtkOpenGLClearErrorMacro();
  vtkOpenGLExtensionManager *extensions = openGLRenderWindow->GetExtensionManager();
  extensions->LoadExtension("GL_ARB_framebuffer_object");
  vtkOpenGLCheckErrorMacro("after extension load");

  this->Initialized = true;
}


//-----------------------------------------------------------------------------
bool vtkOpenGLShaderComputation::AcquireFramebuffer()
{
  //
  // adapted from
  // https://www.opengl.org/wiki/Framebuffer_Object_Examples
  //

  int resultDimensions[3];
  this->ResultImageData->GetDimensions(resultDimensions);

  vtkOpenGLClearErrorMacro();

  //
  // generate and bind our Framebuffer
  vtkgl::GenFramebuffers(1, &(this->FramebufferID));
  vtkgl::BindFramebuffer(vtkgl::FRAMEBUFFER, this->FramebufferID);

  //
  // Create and attach a color buffer
  // * We must bind this->ColorRenderbufferID before we call glRenderbufferStorage
  // * The storage format is RGBA8
  // * Attach color buffer to FBO
  //
  vtkgl::GenRenderbuffers(1, &(this->ColorRenderbufferID));
  vtkgl::BindRenderbuffer(vtkgl::RENDERBUFFER, this->ColorRenderbufferID);
  vtkgl::RenderbufferStorage(vtkgl::RENDERBUFFER, GL_RGBA8,
                           resultDimensions[0], resultDimensions[1]);
  vtkgl::FramebufferRenderbuffer(vtkgl::FRAMEBUFFER,
                               vtkgl::COLOR_ATTACHMENT0,
                               vtkgl::RENDERBUFFER,
                               this->ColorRenderbufferID);

  //
  // Now do the same for the depth buffer
  //
  vtkgl::GenRenderbuffers(1, &(this->DepthRenderbufferID));
  vtkgl::BindRenderbuffer(vtkgl::RENDERBUFFER, this->DepthRenderbufferID);
  vtkgl::RenderbufferStorage(vtkgl::RENDERBUFFER, GL_DEPTH_COMPONENT24,
                           resultDimensions[0], resultDimensions[1]);
  vtkgl::FramebufferRenderbuffer(vtkgl::FRAMEBUFFER,
                               vtkgl::DEPTH_ATTACHMENT,
                               vtkgl::RENDERBUFFER,
                               this->DepthRenderbufferID);

  //
  // Does the GPU support current Framebuffer configuration?
  //
  GLenum status;
  status = vtkgl::CheckFramebufferStatus(vtkgl::FRAMEBUFFER);
  switch(status)
    {
    case vtkgl::FRAMEBUFFER_COMPLETE:
      break;
    default:
      vtkOpenGLCheckErrorMacro("after bad framebuffer status");
      vtkErrorMacro("Bad framebuffer configuration, status is: " << status);
      return false;
    }

  //
  // now we can render to the FBO (also called RenderBuffer)
  //
  vtkgl::BindFramebuffer(vtkgl::FRAMEBUFFER, this->FramebufferID);
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  //
  // Set up a normalized rendering environment
  //
  glViewport(0, 0, resultDimensions[0], resultDimensions[1]);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, resultDimensions[0], 0.0, resultDimensions[1], -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);

  vtkOpenGLCheckErrorMacro("after framebuffer acquisition");
  return true;
}

//----------------------------------------------------------------------------
void vtkOpenGLShaderComputation::ReleaseFramebuffer()
{
  vtkOpenGLClearErrorMacro();
  //Delete temp resources
  vtkgl::DeleteRenderbuffers(1, &(this->ColorRenderbufferID));
  vtkgl::DeleteRenderbuffers(1, &(this->DepthRenderbufferID));
  //Bind 0, which means render to back buffer, as a result, this->FramebufferID is unbound
  vtkgl::BindFramebuffer(vtkgl::FRAMEBUFFER, 0);
  vtkgl::DeleteFramebuffers(1, &(this->FramebufferID));
  vtkOpenGLCheckErrorMacro("after framebuffer release");
}

//----------------------------------------------------------------------------
// Perform the computation
//
void vtkOpenGLShaderComputation::Compute()
{

  // bail out early if we aren't configured corretly
  if (this->VertexShaderSource == NULL || this->FragmentShaderSource == NULL)
    {
    vtkErrorMacro("Both vertex and fragment shaders are needed for a shader computation.");
    return;
    }

  // check and set up the result area
  if (this->ResultImageData == NULL
      ||
      this->ResultImageData->GetPointData() == NULL
      ||
      this->ResultImageData->GetPointData()->GetScalars() == NULL
      ||
      this->ResultImageData->GetPointData()->GetScalars()->GetVoidPointer(0) == NULL)
    {
    vtkErrorMacro("Result image data is not correctly set up.");
    return;
    }
  int resultDimensions[3];
  this->ResultImageData->GetDimensions(resultDimensions);
  vtkPointData *pointData = this->ResultImageData->GetPointData();
  vtkDataArray *scalars = pointData->GetScalars();
  void *resultPixels = scalars->GetVoidPointer(0);

  // ensure that all our OpenGL calls go to the correct context
  this->RenderWindow->MakeCurrent();

  if (!this->AcquireFramebuffer()) // TODO: should re-use the framebuffer for efficiency
    {
    vtkErrorMacro("Could not set up a framebuffer.");
    return;
    }

  // Configure the program and the input data
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

  // define a normalized computing surface
  GLfloat planeVertices[] = { -1.0f, -1.0f, 0.0f,
                              -1.0f,  1.0f, 0.0f,
                               1.0f,  1.0f, 0.0f,
                               1.0f, -1.0f, 0.0f
                        };
  GLuint planeVerticesSize = sizeof(GLfloat)*3*4;
  GLfloat planeTextureCoordinates[] = { 0.0f, 1.0f,
                                        0.0f, 0.0f,
                                        1.0f, 0.0f,
                                        1.0f, 1.0f
                        };
  GLuint planeTextureCoordinatesSize = sizeof(GLfloat)*2*4;

  vtkOpenGLClearErrorMacro();
  // Use the program object
  glUseProgram ( this->ProgramObject );

  // put vertices in a buffer and make it available to the program
  GLuint vertexLocation = glGetAttribLocation(this->ProgramObject, "vertexAttribute");
  GLuint planeVerticesBuffer;
  glGenBuffers(1, &planeVerticesBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, planeVerticesBuffer);
  glBufferData(GL_ARRAY_BUFFER, planeVerticesSize, planeVertices, GL_STATIC_DRAW);
  glEnableVertexAttribArray ( vertexLocation );
  glVertexAttribPointer ( vertexLocation, 3, GL_FLOAT, GL_FALSE, 0, 0 );

  // texture coordinates in a buffer
  GLuint textureCoordinatesLocation = glGetAttribLocation(this->ProgramObject,
                                                          "textureCoordinateAttribute");
  GLuint textureCoordinatesBuffer;
  glGenBuffers(1, &textureCoordinatesBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, textureCoordinatesBuffer);
  glBufferData(GL_ARRAY_BUFFER, planeTextureCoordinatesSize, planeTextureCoordinates, GL_STATIC_DRAW);
  glEnableVertexAttribArray ( textureCoordinatesLocation );
  glVertexAttribPointer ( textureCoordinatesLocation, 2, GL_FLOAT, GL_FALSE, 0, 0 );

  // make sure the texture is bound and pass in the address of it
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, this->TextureID);
  GLuint volumeSamplerLocation = glGetUniformLocation(this->ProgramObject, "volumeSampler");
  glUniform1i(volumeSamplerLocation, 0);

  //
  // GO!
  //
  glDrawArrays ( GL_QUADS, 0, 4 );

  //
  // Collect the results of the calculation back into the image data
  //
  glReadPixels(0, 0, resultDimensions[0], resultDimensions[1], GL_RGBA, GL_UNSIGNED_BYTE, resultPixels);

  vtkOpenGLCheckErrorMacro("after computing and reading back");
  //
  // Don't use the program or the framebuffer anymore
  //
  glUseProgram ( 0 );
  this->ReleaseFramebuffer();
}

//----------------------------------------------------------------------------
void vtkOpenGLShaderComputation::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
  os << indent << "Initialized: " << this->Initialized << "\n";
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
  if ( this->ResultImageData )
    {
    os << indent << "ResultImageData: " << this->ResultImageData << "\n";
    }
  else
    {
    os << indent << "ResultImageData: (none)\n";
    }
  os << indent << "ProgramObject: " << this->ProgramObject << "\n";
  os << indent << "ProgramObjectMTime: " << this->ProgramObjectMTime << "\n";
  os << indent << "TextureID: " << this->TextureID << "\n";
  os << indent << "TextureMTime: " << this->TextureMTime << "\n";
  os << indent << "FramebufferID: " << this->FramebufferID << "\n";
  os << indent << "ColorRenderbufferID: " << this->ColorRenderbufferID << "\n";
  os << indent << "DepthRenderbufferID: " << this->DepthRenderbufferID << "\n";
}
