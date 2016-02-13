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
# GLVolume
#

class GLVolume():
  """The code to represent Slicer scalar volumes"""

  def __init__(self,shaderComputation,textureUnit,volumeNode):
    self.shaderComputation = shaderComputation
    self.textureUnit = textureUnit
    self.volumeNode = volumeNode
    self.update()

  def update(self):
    """to be called when image data has changed"""
    self.image = self.volumeNode.GetImageData()

    self.shiftScale = vtk.vtkImageShiftScale()
    self.shiftScale.SetInputData(self.image)
    self.shiftScale.SetOutputScalarTypeToUnsignedShort()
    low, high = self.image.GetScalarRange()
    self.shiftScale.SetShift(-low)
    if high == low:
      scale = 1.
    else:
      scale = (1. * vtk.VTK_UNSIGNED_SHORT_MAX) / (high-low)
    self.shiftScale.SetScale(scale)
    self.sampleUnshift = low
    self.sampleUnscale = high-low
    self.shiftScale.Update()
    normalizedImage = self.shiftScale.GetOutputDataObject(0)

    self.textureImage=vtkOpenGLTextureImage()
    self.textureImage.SetShaderComputation(self.shaderComputation)
    self.textureImage.SetImageData(normalizedImage)
    self.textureImage.Activate(self.textureUnit)
    print('activated volume %s in textureUnit %d' % (self.volumeNode.GetName(), self.textureUnit))

  def keys(self):
    return {
      'sampleUnshift%d' % self.textureUnit : self.sampleUnshift,
      'sampleUnscale%d' % self.textureUnit : self.sampleUnscale,
    }


#
# GLFilter
#

class GLFilter():
  """The general code for filters applied to Slicer scalar volumes"""

  def __init__(self,volumes,vertexShaderTemplate,fragmentShaderTemplate):
    self.showImageViewer = True
    self.vertexShaderTemplate = vertexShaderTemplate
    self.fragmentShaderTemplate = fragmentShaderTemplate

    # make a result image to display
    self.resultImage = vtk.vtkImageData()
    self.resultImage.SetDimensions((512,512,1))
    self.resultImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)

    #
    # Shader computation
    # - need to import class from module here since it may not be in sys.path
    #   at startup time
    # - uses dummy render window for framebuffer object context
    #
    self.shaderComputation=vtkOpenGLShaderComputation()
    self.glVolumes = []
    for volume in volumes:
      textureUnit = len(self.glVolumes)
      self.glVolumes.append(GLVolume(self.shaderComputation, textureUnit, volume))
      print('volume %s in textureUnit %d' % (volume.GetName(), textureUnit))
    self.shaderComputation.SetResultImageData(self.resultImage)
    self.shaderComputation.AcquireResultRenderbuffer()

  def compute(self,keys):
    for glVolume in self.glVolumes:
      keys.update(glVolume.keys())
    self.shaderComputation.SetVertexShaderSource(self.vertexShaderTemplate % keys)
    self.shaderComputation.SetFragmentShaderSource(self.fragmentShaderTemplate % keys)
    self.shaderComputation.Compute()
    self.shaderComputation.ReadResult()

    if self.showImageViewer:
      if not hasattr(self,'imageViewer'):
        self.imageViewer = vtk.vtkImageViewer()
        self.imageViewer.SetPosition(20, 500)
        self.imageViewer.SetColorWindow(256)
        self.imageViewer.SetColorLevel(128)
      self.imageViewer.SetInputData(self.resultImage)
      self.imageViewer.Render()


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

  def thresholdFilter(self, inputVolume, threshold):

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
      float textureSampleDenormalized(const in sampler3D volumeTextureUnit,
                                      const in vec3 stpPoint) {
        return ( texture3D(volumeTextureUnit, stpPoint).r * %(sampleUnscale0)f
                                                          + %(sampleUnshift0)f );
      }

      varying vec3 interpolatedTextureCoordinate;
      uniform sampler3D textureUnit0; // input
      void main()
      {
        vec3 samplePoint = vec3(interpolatedTextureCoordinate);
        float sample = textureSampleDenormalized(textureUnit0,
                                           vec3(samplePoint.xy, %(slice)f));
        gl_FragColor = vec4(1., 1., 0., 1.);
        if (sample > %(threshold)f) {
          gl_FragColor = vec4(1., 0., 0., 1.);
        }
      }
    """

    keys = {
      'threshold' : threshold + 100,
    }

    filter = GLFilter([inputVolume], vertexShaderTemplate, fragmentShaderTemplate)
    filter.showImageViewer = True

    inputImage = inputVolume.GetImageData()
    slices = inputImage.GetDimensions()[2]

    for slice in range(slices):
      keys['slice'] = slice / (1. * slices)
      filter.compute(keys)
    return filter

  def smoothFilter(self, inputVolume, outputVolume, sigma):

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
      uniform sampler3D textureUnit0; // input
      uniform sampler3D textureUnit1; // output
      void main()
      {
        vec3 samplePoint = vec3(interpolatedTextureCoordinate.xy, %(slice)f);
        float normalizedSample = texture3D(textureUnit1, samplePoint).r;

        gl_FragColor = vec4(vec3(normalizedSample), 1.);
      }
    """

    keys = {
      'sigma' : sigma,
    }

    filter = GLFilter([inputVolume, outputVolume], vertexShaderTemplate, fragmentShaderTemplate)
    filter.showImageViewer = True

    inputImage = inputVolume.GetImageData()
    slices = inputImage.GetDimensions()[2]

    for slice in range(slices):
      keys['slice'] = slice / (1. * slices)
      filter.compute(keys)
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
    outputImage = vtk.vtkImageData()
    outputImage.SetDimensions(inputImage.GetDimensions())
    outputImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)

    # make output match input geometry
    rasToIJK = vtk.vtkMatrix4x4()
    inputVolume.GetRASToIJKMatrix(rasToIJK)
    #outputVolume.SetAndObserveImageData(outputImage)
    outputVolume.SetRASToIJKMatrix(rasToIJK)
    outputVolume.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())

    # get the opengl context and set it up with the two textures

    shaderComputation=vtkOpenGLShaderComputation()

    self.inputTextureImage=vtkOpenGLTextureImage()
    self.inputTextureImage.SetShaderComputation(shaderComputation)
    self.inputTextureImage.SetImageData(normalizedImage)
    self.inputTextureImage.Activate(2)

    outputTextureImage=vtkOpenGLTextureImage()
    outputTextureImage.SetShaderComputation(shaderComputation)
    outputTextureImage.SetImageData(outputImage)
    outputTextureImage.Activate(1)

    shaderComputation.SetResultImageData(outputImage)
    shaderComputation.AcquireResultRenderbuffer()

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
      float textureSampleDenormalized(const in sampler3D volumeTextureUnit,
                                      const in vec3 stpPoint) {
        return ( texture3D(volumeTextureUnit, stpPoint).r * %(sampleUnscale)f
                                                          + %(sampleUnshift)f );
      }

      varying vec3 interpolatedTextureCoordinate;
      uniform sampler3D textureUnit2; // input
      void main()
      {
        vec3 samplePoint = vec3(interpolatedTextureCoordinate);
        float sample = textureSampleDenormalized(textureUnit2,
                                           vec3(samplePoint.xy, .5));
        float normalizedSample = (sample - %(sampleUnshift)f) / %(sampleUnscale)f;


        normalizedSample = texture3D(textureUnit2,
                                     vec3(interpolatedTextureCoordinate.xy,%(slice)f)).r;

        gl_FragColor = vec4(normalizedSample, vec3(1.)) +
                  vec4(sin(%(slice)f * 100.*interpolatedTextureCoordinate), 1.);
      }
    """

    keys = {
      'sampleUnshift' : sampleUnshift,
      'sampleUnscale' : sampleUnscale,
    }
    slices = inputImage.GetDimensions()[2]
    for slice in range(slices):
      keys['slice'] = slice / (1. * slices)
      shaderComputation.SetVertexShaderSource(vertexShaderSourceTemplate % keys)
      shaderComputation.SetFragmentShaderSource(fragmentShaderSourceTemplate % keys)
      outputTextureImage.AttachAsDrawTarget(0, slice)
      shaderComputation.Compute()

    print(keys)

    outputTextureImage.ReadBack()

    extractComponents = vtk.vtkImageExtractComponents()
    extractComponents.SetInputData(outputImage)
    extractComponents.Update()
    outputVolume.SetAndObserveImageData(extractComponents.GetOutputDataObject(0))


  def run(self, inputVolume, outputVolume, imageThreshold):
    """
    Run the actual algorithm
    """
    import time
    startTime = time.clock()
    logging.info('Processing started')

    #self.smooth(inputVolume, outputVolume, imageThreshold)

    #self.filter = self.thresholdFilter(inputVolume, imageThreshold)
    self.filter = self.smoothFilter(inputVolume, outputVolume, imageThreshold)

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
