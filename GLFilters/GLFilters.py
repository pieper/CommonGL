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
    self.imageThresholdSliderWidget.singleStep = 0.01
    self.imageThresholdSliderWidget.minimum = 0
    self.imageThresholdSliderWidget.maximum = .1
    self.imageThresholdSliderWidget.value = 0.01
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
      TODO: this could factor out into a helper file
      TODO: multiple input filter examples
      TODO: iterative filters with scratch textures
      TODO: compute directly into a texture being rendered by SceneRenderer
  """

  def __init__(self, inputVolumes, outputVolume):

    if len(inputVolumes) < 1:
      raise "Must have at least one input volume"
    self.volume0 = inputVolumes[0]
    if not self.volume0.GetImageData():
      raise "Must have a valid input volume with image data"

    self.outputVolume = outputVolume

    self.shaderComputation=vtkOpenGLShaderComputation()

    import ShaderComputation
    self.volumeTextures = []
    for volume in inputVolumes:
      textureUnit = len(self.volumeTextures)
      volumeTexture = ShaderComputation.VolumeTexture(
                                          self.shaderComputation,
                                          textureUnit,
                                          volume)
      self.volumeTextures.append(volumeTexture)

    # prepare output
    rasToIJK = vtk.vtkMatrix4x4()
    self.volume0.GetRASToIJKMatrix(rasToIJK)
    self.outputVolume.SetRASToIJKMatrix(rasToIJK)
    self.outputVolume.SetAndObserveTransformNodeID(self.volume0.GetTransformNodeID())

    self.iterationVolumeTextures = []
    for iteration in range(2):
      iterationImage = vtk.vtkImageData()
      iterationImage.SetDimensions(self.volume0.GetImageData().GetDimensions())
      iterationImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
      # use the outputVolume for both iteration images
      # so they have the same transforms, lookup tables, etc.
      self.outputVolume.SetAndObserveImageData(iterationImage)
      textureUnit = len(self.volumeTextures)
      iterationVolumeTexture = ShaderComputation.VolumeTexture(
                                          self.shaderComputation,
                                          textureUnit,
                                          outputVolume,
                                          optimizeDynamicRange=False)
      self.iterationVolumeTextures.append(iterationVolumeTexture)
    self.shaderComputation.SetResultImageData(iterationImage)
    self.shaderComputation.AcquireResultRenderbuffer()

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
    textureUnitCount = len(self.volumeTextures) + len(self.iterationVolumeTextures)
    for textureUnit in range(textureUnitCount):
      self.header += "textureUnit%d," % textureUnit
    self.header = self.header[:-1] + ';'

  def iteration(self, targetTextureImage):
    """Perform one whole-volume iteration of the algorithm into a target"""
    slices = self.volume0.GetImageData().GetDimensions()[2]
    for slice_ in range(slices):
      # draw into output texture TODO: move into VolumeTexture class
      targetTextureImage.AttachAsDrawTarget(0, slice_)
      # activate the textures
      for volumeTexture in self.volumeTextures:
        volumeTexture.textureImage.Activate(volumeTexture.textureUnit)
      # perform the computation for this slice
      self.shaderComputation.Compute(slice_ / (1. * slices))

  def compute(self, vertexShader, fragmentShader, keys={}, iterations=1):
    """Perform an iterated filter"""
    import time
    startTime = time.time()
    logging.info('Processing started')

    logging.info('%f shaders set' % (time.time() - startTime))

    for iteration in range(iterations):
      # build the source
      keys['iteration'] = iteration
      if iteration == 0:
        keys['iterationTextureUnit'] = 'textureUnit0'
      else:
        keys['iterationTextureUnit'] = 'textureUnit' + str((iteration+1)%2)
      self.shaderComputation.SetVertexShaderSource(vertexShader % keys)
      samplersSource = ''
      for volumeTexture in self.volumeTextures:
        samplersSource += volumeTexture.fieldSampleSource()
      shaders = self.header + samplersSource + fragmentShader
      self.shaderComputation.SetFragmentShaderSource(shaders % keys)
      # set the target
      targetVolumeTexture = self.iterationVolumeTextures[iteration%2]
      targetTextureImage = targetVolumeTexture.textureImage
      self.iteration(targetTextureImage)
      logging.info('%f iteration %d' % (time.time() - startTime, iteration))

    targetTextureImage.ReadBack()

    logging.info('%f readback finished' % (time.time() - startTime))

    extractComponents = vtk.vtkImageExtractComponents()
    extractComponents.SetInputData(targetTextureImage.GetImageData())
    extractComponents.Update()
    self.outputVolume.SetAndObserveImageData(extractComponents.GetOutputDataObject(0))

    logging.info('%f computation finished' % (time.time() - startTime))
    logging.info('')

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

  def smoothFilter(self, inputVolume, outputVolume, sigma):

    filter_ = GLFilter([inputVolume,], outputVolume)

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
      uniform float slice;
      varying vec3 interpolatedTextureCoordinate;

      void main()
      {
        vec3 samplePoint = vec3(interpolatedTextureCoordinate.xy,slice);

        vec4 sum = vec4(0.);
        for (int offsetX = -%(kernelSize)d; offsetX <= %(kernelSize)d; offsetX++) {
          for (int offsetY = -%(kernelSize)d; offsetY <= %(kernelSize)d; offsetY++) {
            for (int offsetZ = -%(kernelSize)d; offsetZ <= %(kernelSize)d; offsetZ++) {
              vec3 offset = %(kernelSpacing)f * vec3(offsetX, offsetY, offsetZ);
              vec4 sample = texture3D(%(iterationTextureUnit)s, samplePoint + offset);
              sum += sample;
            }
          }
        }

        float sampleCount = pow(2. * (%(kernelSize)f + 1.), 3.);
        gl_FragColor = vec4(vec3(sum/sampleCount), 1.);
      }
    """

    keys = {
      'kernelSize' : 5,
      'kernelSpacing' : sigma,
    }

    filter_.compute(vertexShaderTemplate, fragmentShaderTemplate, keys, 10)

    return filter_

  def run(self, inputVolume, outputVolume, imageThreshold):
    """
    Run the actual algorithm
    """
    self.filter_ = self.smoothFilter(inputVolume, outputVolume, imageThreshold)

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

    inputVolume = slicer.util.getNode(pattern="FA")

    volumesLogic = slicer.modules.volumes.logic()
    outputVolume = volumesLogic.CloneVolume(slicer.mrmlScene, inputVolume, "FA-filtered")
    logic = GLFiltersLogic()
    self.assertIsNotNone( outputVolume )
    logic.run(inputVolume, outputVolume, 0.05)
    self.delayDisplay('Test passed!')
