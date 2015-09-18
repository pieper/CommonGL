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
  this->ResultImageData = NULL;
  this->ProgramObject = 0;
  this->ProgramObjectMTime = 0;

  this->RenderWindow = vtkRenderWindow::New();
  this->RenderWindow->SetOffScreenRendering(1);
  this->Initialize(this->RenderWindow);
}

//----------------------------------------------------------------------------
vtkOpenGLShaderComputation::~vtkOpenGLShaderComputation()
{
  this->SetVertexShaderSource(NULL);
  this->SetFragmentShaderSource(NULL);
  this->SetResultImageData(NULL);
  if (this->ProgramObject > 0)
    {
    glDeleteProgram ( this->ProgramObject );
    this->ProgramObject = 0;
    }
  this->SetRenderWindow(NULL);
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
  vtkOpenGLCheckErrorMacro("after binding framebuffer");

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
  vtkOpenGLCheckErrorMacro("after binding color renderbuffer");

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
  vtkOpenGLCheckErrorMacro("after binding depth renderbuffer");

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
  vtkOpenGLCheckErrorMacro("after clearing renderbuffers");

  //
  // Set up a normalized rendering environment
  //
  glViewport(0, 0, resultDimensions[0], resultDimensions[1]);
  vtkOpenGLCheckErrorMacro("after 1 normalizing environment");
  glMatrixMode(GL_PROJECTION);
  vtkOpenGLCheckErrorMacro("after 2 normalizing environment");
  glLoadIdentity();
  vtkOpenGLCheckErrorMacro("after 3 normalizing environment");
  glOrtho(0.0, resultDimensions[0], 0.0, resultDimensions[1], -1.0, 1.0);
  vtkOpenGLCheckErrorMacro("after 4 normalizing environment");
  glMatrixMode(GL_MODELVIEW);
  vtkOpenGLCheckErrorMacro("after 5 normalizing environment");
  glLoadIdentity();
  vtkOpenGLCheckErrorMacro("after 6 normalizing environment");
  glDisable(GL_BLEND);
  vtkOpenGLCheckErrorMacro("after 8 normalizing environment");
  glEnable(GL_DEPTH_TEST);
  vtkOpenGLCheckErrorMacro("after 9 normalizing environment");

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

  // define a normalized computing surface
  GLfloat planeVertices[] = { -1.0f, -1.0f, 0.0f,
                              -1.0f,  1.0f, 0.0f,
                               1.0f,  1.0f, 0.0f,
                               1.0f, -1.0f, 0.0f
                        };
  GLuint planeVerticesSize = sizeof(GLfloat)*3*4;
  GLfloat planeTextureCoordinates[] = { 0.0f, 0.0f,
                                        0.0f, 1.0f,
                                        1.0f, 1.0f,
                                        1.0f, 0.0f
                        };
  GLuint planeTextureCoordinatesSize = sizeof(GLfloat)*2*4;

  vtkOpenGLClearErrorMacro();
  // Use the program object
  glUseProgram ( this->ProgramObject );
  vtkOpenGLCheckErrorMacro("after use program");

  // put vertices in a buffer and make it available to the program
  GLuint vertexLocation = glGetAttribLocation(this->ProgramObject, "vertexAttribute");
  GLuint planeVerticesBuffer;
  glGenBuffers(1, &planeVerticesBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, planeVerticesBuffer);
  glBufferData(GL_ARRAY_BUFFER, planeVerticesSize, planeVertices, GL_STATIC_DRAW);
  glEnableVertexAttribArray ( vertexLocation );
  glVertexAttribPointer ( vertexLocation, 3, GL_FLOAT, GL_FALSE, 0, 0 );
  vtkOpenGLCheckErrorMacro("after vertices");

  // texture coordinates in a buffer
  GLuint textureCoordinatesLocation = glGetAttribLocation(this->ProgramObject,
                                                          "textureCoordinateAttribute");
  GLuint textureCoordinatesBuffer;
  glGenBuffers(1, &textureCoordinatesBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, textureCoordinatesBuffer);
  glBufferData(GL_ARRAY_BUFFER, planeTextureCoordinatesSize, planeTextureCoordinates, GL_STATIC_DRAW);
  glEnableVertexAttribArray ( textureCoordinatesLocation );
  glVertexAttribPointer ( textureCoordinatesLocation, 2, GL_FLOAT, GL_FALSE, 0, 0 );
  vtkOpenGLCheckErrorMacro("after texture coordinates");

  // Iterate through all standard texture units and if one of them
  // is used as a uniform variable in the program, set the corresponding value.
  // This relies on vtkOpenGLTextureImage (or something else) to have
  // set up the texture units with data.
  // Up to 48 units are meant to be supported on any OpenGL implementation
  // but the defined enums appear to only go to 32.
  #define __TEXTURE_UNIT_COUNT 16 // TODO: maybe expose parameter of how many textures to look for
  char textureUnitUniformString[14]; // 14 length of "textureUnit__" including \0
  strncpy(textureUnitUniformString, "textureUnit__", 14);
  char textureUnitLength = 11; // Up to the two underscores that will be replaced
  char asciiUnit[3]; // target for snprintf
  int unitIndex; 
  for (unitIndex = 0; unitIndex < __TEXTURE_UNIT_COUNT; unitIndex++)
    {
    snprintf(asciiUnit, 3, "%d", unitIndex);
    strncpy(textureUnitUniformString + textureUnitLength, asciiUnit, 2);
    GLint textureUnitSamplerLocation = glGetUniformLocation(this->ProgramObject, textureUnitUniformString);
    if ( textureUnitSamplerLocation >= 0 ) 
      {
      glUniform1i(textureUnitSamplerLocation, unitIndex);
      vtkOpenGLCheckErrorMacro("after setting texture unit uniform " << unitIndex);
      }
    }
  vtkOpenGLCheckErrorMacro("after setting texture unit uniforms");

  //
  // GO!
  //
  glDrawArrays ( GL_QUADS, 0, 4 );

  vtkOpenGLCheckErrorMacro("after drawing");

  //
  // Collect the results of the calculation back into the image data
  //
  glReadPixels(0, 0, resultDimensions[0], resultDimensions[1], GL_RGBA, GL_UNSIGNED_BYTE, resultPixels);
  pointData->Modified();

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
  os << indent << "FramebufferID: " << this->FramebufferID << "\n";
  os << indent << "ColorRenderbufferID: " << this->ColorRenderbufferID << "\n";
  os << indent << "DepthRenderbufferID: " << this->DepthRenderbufferID << "\n";
}
