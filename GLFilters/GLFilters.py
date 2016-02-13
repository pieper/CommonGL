import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging


try:
  from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLShaderComputation
  from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLTextureImage
except ImportError:
  import vtkAddon
  vtkOpenGLShaderComputation=vtkAddon.vtkOpenGLShaderComputation
  vtkOpenGLTextureImage=vtkAddon.vtkOpenGLTextureImage

#
# GLFilters
#

class GLFilters(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "GLFilters" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Filtering"]
    self.parent.dependencies = []
    self.parent.contributors = ["Steve Pieper (Isomics, Inc.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    This is an example of using the vtkOpenGLShaderComputation class to perform
    some cool computation.
    """
    self.parent.acknowledgementText = """
    This file was developed by Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.

""" # replace with organization, grant and thanks.

#
# GLFiltersWidget
#

class GLFiltersWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    self.logic = GLFiltersLogic()

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # output volume selector
    #
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = True
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene( slicer.mrmlScene )
    self.outputSelector.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume: ", self.outputSelector)

    #
    # threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 0.1
    self.imageThresholdSliderWidget.minimum = -100
    self.imageThresholdSliderWidget.maximum = 100
    self.imageThresholdSliderWidget.value = 0.5
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    parametersFormLayout.addRow("Image threshold", self.imageThresholdSliderWidget)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.imageThresholdSliderWidget.connect("valueChanged(double)", self.onApplyButton)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode()

  def onApplyButton(self):
    imageThreshold = self.imageThresholdSliderWidget.value
    self.logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold)

#
# GLFilter
#

class GLFilter(object):
  """ Implement volume filtering with GLSL for mrml nodes
  """

  def __init__(self, volumes, resultImage = None):
    self.resultImage = resultImage
    self.showImageViewer = False

    self.shaderComputation=vtkOpenGLShaderComputation()

    import ShaderComputation
    self.volumeTextures = []
    for volume in volumes:
      textureUnit = len(self.volumeTextures)
      volumeTexture = ShaderComputation.VolumeTexture(
                                          self.shaderComputation,
                                          textureUnit,
                                          volume)
      self.volumeTextures.append(volumeTexture)

    if not self.resultImage:
      self.resultImage = vtk.vtkImageData()
      self.resultImage.SetDimensions(512, 512, 1)
      self.resultImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    self.shaderComputation.SetResultImageData(self.resultImage)

    self.imageViewer = vtk.vtkImageViewer()
    self.imageViewer.SetColorLevel(128)
    self.imageViewer.SetColorWindow(256)
    self.imageViewer.SetPosition(20, 500)

    self.header = """
      #version 120

      vec3 transformPoint(const in vec3 samplePoint)
      {
        return samplePoint; // identity
      }
    """

    # need to declare each texture unit as a uniform passed in
    # from the host code; these are done in the vtkOpenGLTextureImage instances
    self.header += "uniform sampler3D "
    for volumeTexture in self.volumeTextures:
      self.header += "textureUnit%d," % volumeTexture.textureUnit
    self.header = self.header[:-1] + ';'

    self.shaderComputation.AcquireResultRenderbuffer()


  def compute(self,vertexShader, fragmentShader):

    samplersSource = ''
    for volumeTexture in self.volumeTextures:
      volumeTexture.updateFromMRML()
      samplersSource += volumeTexture.fieldSampleSource()

    self.shaderComputation.SetVertexShaderSource(vertexShader)
    shaders = self.header + samplersSource + fragmentShader
    self.shaderComputation.SetFragmentShaderSource(shaders)

    self.shaderComputation.Compute()
    self.shaderComputation.ReadResult()
    #self.shaderComputation.ReleaseResultRenderbuffer()

    if self.showImageViewer:
      self.imageViewer.SetInputData(self.resultImage)
      self.imageViewer.Render()

    if False:
      fp = open('/tmp/shader.glsl','w')
      fp.write(self.shaderComputation.GetFragmentShaderSource())
      fp.close()


#
# GLFiltersLogic
#

class GLFiltersLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    self.showImageViewer = False

  def smoothFilter(self, inputVolume, outputVolume, sigma):

    # make the image data for the output volume
    inputImage = inputVolume.GetImageData()
    outputImage = vtk.vtkImageData()
    outputImage.SetDimensions(inputImage.GetDimensions())
    outputImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    #outputImage.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
    outputVolume.SetAndObserveImageData(outputImage)

    # make output match input geometry
    rasToIJK = vtk.vtkMatrix4x4()
    inputVolume.GetRASToIJKMatrix(rasToIJK)
    outputVolume.SetRASToIJKMatrix(rasToIJK)
    outputVolume.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())

    filter = GLFilter([inputVolume, outputVolume], resultImage=outputImage)
    filter.showImageViewer = True

    vertexShaderTemplate = """
      #version 120
      attribute vec3 vertexAttribute;
      attribute vec2 textureCoordinateAttribute;
      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        interpolatedTextureCoordinate = vec3(textureCoordinateAttribute, .5);
        gl_Position = vec4(vertexAttribute, 1.);
      }
    """

    fragmentShaderTemplate = """
      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        vec3 samplePoint = vec3(interpolatedTextureCoordinate);
        float sample = textureSampleDenormalized0(textureUnit0,
                                           vec3(samplePoint.xy, .5));

        float normalizedSample = texture3D(textureUnit0,
                             vec3(interpolatedTextureCoordinate.xy,%(slice)f)).r;

        gl_FragColor = vec4(vec3(normalizedSample), 1.);
      }
    """

    outputTexture = filter.volumeTextures[1].textureImage
    keys = {}
    slices = inputImage.GetDimensions()[2]
    for slice in range(slices):
      keys['slice'] = slice / (1. * slices)
      #outputTexture.AttachAsDrawTarget(0, slice)
      filter.compute(vertexShaderTemplate % keys, fragmentShaderTemplate % keys)

    print('reading back')
    outputTexture.ReadBack()

    print('extracting')
    extractComponents = vtk.vtkImageExtractComponents()
    extractComponents.SetInputData(outputImage)
    extractComponents.Update()
    outputVolume.SetAndObserveImageData(extractComponents.GetOutputDataObject(0))

    return filter

  def smooth(self, inputVolume, outputVolume, sigma):

    # since the OpenGL texture will be floats in the range 0 to 1, all negative values
    # will get clamped to zero.  Also if the sample values aren't evenly spread through
    # the zero to one space we may run into numerical issues.  So rescale the data to the
    # to fit in the full range of the a 16 bit short.
    # (any vtkImageData scalar type should work with this approach)
    # TODO: move this to the vtkOpenGLTextureImage class
    inputImage = inputVolume.GetImageData()
    if not hasattr(self,'shiftScale'):
      self.shiftScale = vtk.vtkImageShiftScale()
    self.shiftScale.SetInputData(inputImage)
    self.shiftScale.SetOutputScalarTypeToUnsignedShort()
    low, high = inputImage.GetScalarRange()
    self.shiftScale.SetShift(-low)
    if high == low:
      scale = 1.
    else:
      scale = (1. * vtk.VTK_UNSIGNED_SHORT_MAX) / (high-low)
    self.shiftScale.SetScale(scale)
    sampleUnshift = low
    sampleUnscale = high-low
    self.shiftScale.Update()
    normalizedImage = self.shiftScale.GetOutputDataObject(0)

    # make the image data for the output volume
    if not outputVolume.GetImageData():
      outputImage = vtk.vtkImageData()
      outputImage.SetDimensions(inputImage.GetDimensions())
      outputImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
      outputVolume.SetAndObserveImageData(outputImage)
    outputImage = outputVolume.GetImageData()

    # make output match input geometry
    rasToIJK = vtk.vtkMatrix4x4()
    inputVolume.GetRASToIJKMatrix(rasToIJK)
    outputVolume.SetRASToIJKMatrix(rasToIJK)
    outputVolume.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())

    # get the opengl context and set it up with the two textures

    shaderComputation=vtkOpenGLShaderComputation()

    resultImage = vtk.vtkImageData()
    columns, rows, slices = outputImage.GetDimensions()
    resultImage.SetDimensions(columns, rows, slices)
    resultImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    shaderComputation.SetResultImageData(resultImage)
    shaderComputation.AcquireResultRenderbuffer()

    outputTextureImage=vtkOpenGLTextureImage()
    outputTextureImage.SetShaderComputation(shaderComputation)
    outputTextureImage.SetImageData(resultImage)

    inputTextureImage=vtkOpenGLTextureImage()
    inputTextureImage.SetShaderComputation(shaderComputation)
    inputTextureImage.SetImageData(normalizedImage)
    inputTextureImage.Activate(10)

    vertexShaderSourceTemplate = """
      #version 120
      attribute vec3 vertexAttribute;
      attribute vec2 textureCoordinateAttribute;
      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        interpolatedTextureCoordinate = vec3(textureCoordinateAttribute, .5);
        gl_Position = vec4(vertexAttribute, 1.);
      }
    """

    fragmentShaderSourceTemplate = """
      varying vec3 interpolatedTextureCoordinate;
      uniform sampler3D textureUnit10; // input
      void main()
      {
        vec3 samplePoint = vec3(interpolatedTextureCoordinate.xy,%(slice)f);
        float normalizedSample;

        normalizedSample = 0.5 * texture3D(textureUnit10, samplePoint).r;

        vec4 sum = vec4(0.);
        for (int offsetX = -%(kernelSize)d; offsetX <= %(kernelSize)d; offsetX++) {
          for (int offsetY = -%(kernelSize)d; offsetY <= %(kernelSize)d; offsetY++) {
            vec3 offset = %(kernelSpacing)f * vec3(offsetX, offsetY, 0);
            vec4 sample = texture3D(textureUnit10, samplePoint + offset);
            sum += sample;
          }
        }

        float sampleCount = pow(2. * (%(kernelSize)f + 1.), 2.);
        gl_FragColor = vec4(vec3(sum/sampleCount), 1.);
      }
    """

    keys = {
      'sampleUnshift' : sampleUnshift,
      'sampleUnscale' : sampleUnscale,
      'kernelSize' : 3,
      'kernelSpacing' : .02,
    }
    slices = inputImage.GetDimensions()[2]
    for slice in range(slices):
      keys['slice'] = slice / (1. * slices)
      shaderComputation.SetVertexShaderSource(vertexShaderSourceTemplate % keys)
      shaderComputation.SetFragmentShaderSource(fragmentShaderSourceTemplate % keys)
      outputTextureImage.AttachAsDrawTarget(0, slice)
      inputTextureImage.Activate(10)
      shaderComputation.Compute()
      shaderComputation.ReadResult()

      if False or self.showImageViewer:
        if not hasattr(self,'imageViewer'):
          self.imageViewer = vtk.vtkImageViewer()
          self.imageViewer.SetPosition(20, 500)
          self.imageViewer.SetColorWindow(256)
          self.imageViewer.SetColorLevel(128)
        self.imageViewer.SetInputData(resultImage)
        self.imageViewer.Render()

    shaderComputation.ReleaseResultRenderbuffer()

    print(keys)

    outputTextureImage.ReadBack()

    extractComponents = vtk.vtkImageExtractComponents()
    extractComponents.SetInputData(resultImage)
    extractComponents.Update()
    outputVolume.SetAndObserveImageData(extractComponents.GetOutputDataObject(0))


  def run(self, inputVolume, outputVolume, imageThreshold):
    """
    Run the actual algorithm
    """
    import time
    startTime = time.clock()
    logging.info('Processing started')

    self.smooth(inputVolume, outputVolume, imageThreshold)
    #self.filter = self.smoothFilter(inputVolume, outputVolume, imageThreshold)

    endTime = time.clock()
    logging.info('%f Processing completed' % (endTime - startTime))

    return True


class GLFiltersTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_GLFilters1()

  def test_GLFilters1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = GLFiltersLogic()
    self.assertIsNotNone( volumeNode )
    self.delayDisplay('Test passed!')
