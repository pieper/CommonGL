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
    self.imageThresholdSliderWidget.singleStep = 0.5
    self.imageThresholdSliderWidget.minimum = 0
    self.imageThresholdSliderWidget.maximum = 5.
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
      TODO: this could factor out into a helper file
      TODO: multiple input filter examples
      TODO: iterative filters with scratch textures
      TODO: compute directly into a texture being rendered by SceneShader
  """

  def __init__(self, inputVolumes, outputVolume):
    """Configure outputVolume and an iterationVolume to match
    the first inputVolume."""

    self.inputVolumes = inputVolumes
    self.outputVolume = outputVolume

    if len(inputVolumes) < 1:
      raise "Must have at least one input volume"
    self.volume0 = inputVolumes[0]
    if not self.volume0.GetImageData():
      raise "Must have a valid input volume with image data"

    # TODO: caller should be required to specify all scratch volumes
    iterationName = '%s-iteration' % self.outputVolume.GetName()
    try:
      self.iterationVolume = slicer.util.getNode(iterationName)
    except slicer.util.MRMLNodeNotFoundException:
      self.iterationVolume = None
    if not self.iterationVolume:
      self.iterationVolume = slicer.vtkMRMLScalarVolumeNode()
      self.iterationVolume.SetName(iterationName)
      slicer.mrmlScene.AddNode(self.iterationVolume)

    rasToIJK = vtk.vtkMatrix4x4()
    self.volume0.GetRASToIJKMatrix(rasToIJK)
    for volume in [self.iterationVolume, self.outputVolume]:
      volume.SetRASToIJKMatrix(rasToIJK)
      volume.SetAndObserveTransformNodeID(self.volume0.GetTransformNodeID())
      image = vtk.vtkImageData()
      image.SetDimensions(self.volume0.GetImageData().GetDimensions())
      image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4) # TODO: needs to be RGBA for rendering
      volume.SetAndObserveImageData(image)

    self.header = """
      #version 150

      vec3 transformPoint(const in vec3 samplePoint)
      {
        return samplePoint; // identity
      }
    """

    self.vertexShaderTemplate = """
      #version 150
      in vec3 vertexCoordinate;
      in vec2 textureCoordinate;
      out vec3 interpolatedTextureCoordinate;
      void main()
      {
        interpolatedTextureCoordinate = vec3(textureCoordinate, .5);
        gl_Position = vec4(vertexCoordinate, 1.);
      }
    """

    self.readBackToVolumeNode = False
    self.dummyImage = vtk.vtkImageData()
    self.dummyImage.SetDimensions(5,5,5)
    self.dummyImage.AllocateScalars(vtk.VTK_SHORT, 1)

  def iteration(self, sceneShader, targetTextureImage):
    """Perform one whole-volume iteration of the algorithm into a target"""
    sceneShader.shaderComputation.SetResultImageData(targetTextureImage.GetImageData())
    sceneShader.shaderComputation.AcquireResultRenderbuffer()
    slices = self.volume0.GetImageData().GetDimensions()[2]
    for slice_ in range(slices):
      # draw into output texture TODO: move into VolumeTexture class
      targetTextureImage.AttachAsDrawTarget(0, slice_)
      # activate the textures
      for volumeTexture in self.volumeTextures:
        if volumeTexture.textureImage != targetTextureImage:
          volumeTexture.textureImage.Activate(volumeTexture.textureUnit)
      # perform the computation for this slice
      sceneShader.shaderComputation.Compute(slice_ / (1. * slices))

  def compute(self, vertexShader, fragmentShader, keys={}, iterations=1):
    """Perform an iterated filter"""

    import time
    startTime = time.time()
    logging.info('-'*40)
    logging.info('Processing started')

    import ShaderComputation
    sceneShader = ShaderComputation.SceneShader.getInstance()
    #sceneShader.resetFieldSamplers()
    sceneShader.updateFieldSamplers()

    volume0Texture = sceneShader.fieldSamplersByNodeID[self.volume0.GetID()]
    outputTextures = {}
    outputTextures[0] = sceneShader.fieldSamplersByNodeID[self.outputVolume.GetID()]
    outputTextures[1] = sceneShader.fieldSamplersByNodeID[self.iterationVolume.GetID()]
    self.volumeTextures = [volume0Texture,] + outputTextures.values()

    logging.info('%f Acquired renderer' % (time.time() - startTime))

    for iteration in range(iterations):
      # build the source
      keys['iteration'] = iteration
      if iteration == 0:
        keys['inputTextureUnit'] = volume0Texture.textureUnit
        keys['inputTextureUnitIdentifier'] = volume0Texture.textureUnitIdentifier()
      else:
        keys['inputTextureUnit'] = outputTextures[(iteration+1)%2].textureUnit
        keys['inputTextureUnitIdentifier'] = outputTextures[(iteration+1)%2].textureUnitIdentifier()
      sceneShader.shaderComputation.SetVertexShaderSource(vertexShader % keys)
      samplersSource = sceneShader.textureUnitDeclarationSource()
      for volumeTexture in self.volumeTextures:
        samplersSource += volumeTexture.fieldSampleSource()
      shaders = self.header + samplersSource + fragmentShader
      sceneShader.shaderComputation.SetFragmentShaderSource(shaders % keys)
      # set the target
      targetTextureImage = outputTextures[iteration%2].textureImage
      self.iteration(sceneShader, targetTextureImage)
      logging.info('%f iteration %d' % (time.time() - startTime, iteration))
      #print(sceneShader.shaderComputation.GetFragmentShaderSource())
      #break

    sceneShader.render()
    if True or self.readBackToVolumeNode:
      targetTextureImage.ReadBack()

      logging.info('%f readback finished' % (time.time() - startTime))

      extractComponents = vtk.vtkImageExtractComponents()
      extractComponents.SetInputData(targetTextureImage.GetImageData())
      extractComponents.Update()
      self.outputVolume.SetAndObserveImageData(extractComponents.GetOutputDataObject(0))
    else:
      sceneShader.render()

    logging.info('%f computation finished' % (time.time() - startTime))
    logging.info('-'*40)

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

  def identityFilter(self, inputVolume, outputVolume, sigma):

    glFilter = GLFilter([inputVolume,], outputVolume)

    fragmentShaderTemplate = """
      uniform float slice;
      in vec3 interpolatedTextureCoordinate;
      out vec4 fragmentColor;

      void main()
      {
        vec3 stpPoint = vec3(interpolatedTextureCoordinate.xy,slice);
        vec3 rasPoint = stpToRAS%(inputTextureUnit)s(stpPoint);
        stpPoint = rasToSTP%(inputTextureUnit)s(rasPoint);

        fragmentColor = vec4( vec3(texture(%(inputTextureUnitIdentifier)s, stpPoint).stp), 1. );
      }
    """

    keys = {
      'kernelSize' : 1,
      'kernelSpacing' : sigma,
    }

    glFilter.compute(glFilter.vertexShaderTemplate, fragmentShaderTemplate, keys, 1)

    return glFilter

  def smoothFilter(self, inputVolume, outputVolume, sigma):

    glFilter = GLFilter([inputVolume,], outputVolume)

    fragmentShaderTemplate = """
      uniform float slice;
      in vec3 interpolatedTextureCoordinate;
      out vec4 fragmentColor;

      void main()
      {
        vec3 stpPoint = vec3(interpolatedTextureCoordinate.xy,slice);
        vec3 rasPoint = stpToRAS%(inputTextureUnit)s(stpPoint);

        float sum = 0.;
        for (int offsetX = -%(kernelSize)d; offsetX <= %(kernelSize)d; offsetX++) {
          for (int offsetY = -%(kernelSize)d; offsetY <= %(kernelSize)d; offsetY++) {
            for (int offsetZ = -%(kernelSize)d; offsetZ <= %(kernelSize)d; offsetZ++) {
              vec3 offset = %(kernelSpacing)f * vec3(offsetX, offsetY, offsetZ);
              vec3 stpSamplePoint = rasToSTP%(inputTextureUnit)s(rasPoint + offset);
              sum += textureSampleDenormalized%(inputTextureUnit)s(%(inputTextureUnitIdentifier)s, stpSamplePoint);
            }
          }
        }

        float sampleCount = pow(2*%(kernelSize)f + 1., 3.);
        float sample = sum/sampleCount;
        float normalizedSample = normalizeSample%(inputTextureUnit)s(sample);
        fragmentColor = vec4(vec3(normalizedSample), 1.);
        //fragmentColor = vec4(vec3(stpPoint.x + stpPoint.y + stpPoint.z), 1.);
        //fragmentColor = vec4(vec3(texture(%(inputTextureUnitIdentifier)s, stpPoint).rgb), 1.);
      }
    """

    keys = {
      'kernelSize' : 1,
      'kernelSpacing' : sigma,
    }

    glFilter.compute(glFilter.vertexShaderTemplate, fragmentShaderTemplate, keys, 1)

    return glFilter

  def run(self, inputVolume, outputVolume, imageThreshold):
    """
    Run the actual algorithm
    """
    #self.glFilter = self.identityFilter(inputVolume, outputVolume, imageThreshold)
    self.glFilter = self.smoothFilter(inputVolume, outputVolume, imageThreshold)

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
    """Test a multi-input filter"""
    import ShaderComputation
    nodes = ShaderComputation.ShaderComputationTest().amigoMRUSPreIntraData()

    outputNode = slicer.vtkMRMLScalarVolumeNode()
    outputNode.SetName('output')
    slicer.mrmlScene.AddNode(outputNode)

    logic = GLFiltersLogic()
    logic.run(nodes[0], outputNode, 1)

  def test_GLFilters2(self):
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
