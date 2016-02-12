import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

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

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode()

  def onApplyButton(self):
    logic = GLFiltersLogic()
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold)

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

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def smooth(self, inputVolume, outputVolume, sigma):

    # make the image data for the output volume
    inputImage = inputVolume.GetImageData()
    outputImage = vtk.vtkImageData()
    outputImage.SetDimensions(inputImage.GetDimensions())
    outputImage.AllocateScalars(inputImage.GetScalarType(), inputImage.GetNumberOfScalarComponents())

    # make output match input geometry
    rasToIJK = vtk.vtkMatrix4x4()
    inputVolume.GetRASToIJKMatrix(rasToIJK)
    outputVolume.SetAndObserveImageData(outputImage)
    outputVolume.SetRASToIJKMatrix(rasToIJK)
    outputVolume.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())

    # get the opengl context and set it up with the two textures
    try:
      # TODO: this branch is when CommonGL is built as extension
      from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLShaderComputation
      from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLTextureImage
    except ImportError:
      # TODO: this branch is when CommonGL is built as part of vtkAddon
      import vtkAddon
      vtkOpenGLShaderComputation=vtkAddon.vtkOpenGLShaderComputation
      vtkOpenGLTextureImage=vtkAddon.vtkOpenGLTextureImage

    shaderComputation=vtkOpenGLShaderComputation()

    inputTextureImage=vtkOpenGLTextureImage()
    inputTextureImage.SetShaderComputation(shaderComputation)
    inputTextureImage.SetImageData(inputImage)
    inputTextureImage.Activate(0)
    
    outputTextureImage=vtkOpenGLTextureImage()
    outputTextureImage.SetShaderComputation(shaderComputation)
    outputTextureImage.SetImageData(outputImage)
    outputTextureImage.Activate(1)

    #shaderComputation.SetResultImageData(outputImage)
    resultImage = vtk.vtkImageData()
    columns,rows,slices = inputImage.GetDimensions()
    resultImage.SetDimensions( (columns, rows, 1) )
    resultImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    #shaderComputation.SetResultImageData(resultImage)
    shaderComputation.SetResultImageData(outputImage)
    shaderComputation.AcquireResultRenderbuffer()

    vertexShaderSourceTemplate = """
      #version 120
      attribute vec3 vertexAttribute;
      attribute vec2 textureCoordinateAttribute;
      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        interpolatedTextureCoordinate = vec3(textureCoordinateAttribute, %(slice)d + .5);
        gl_Position = vec4(vertexAttribute, 1.);
      }
    """

    # TODO: take into account the shift/scale
    shaderComputation.SetFragmentShaderSource("""
      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        gl_FragColor = vec4(interpolatedTextureCoordinate, %(alpha)f);
      }
    """ % {
      'alpha' : 0.8,
    })

    # TODO: iterate and set the target slice plane
    for slice in range(1+0*inputImage.GetDimensions()[2]):
      print('drawing slice', slice)
      shaderComputation.SetVertexShaderSource(vertexShaderSourceTemplate % { 'slice' : slice })
      outputTextureImage.AttachAsDrawTarget(0, slice)
      shaderComputation.Compute()

    outputTextureImage.ReadBack()



  def run(self, inputVolume, outputVolume, imageThreshold):
    """
    Run the actual algorithm
    """

    if not self.isValidInputOutputData(inputVolume, outputVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

    self.smooth(inputVolume, outputVolume, imageThreshold)

    if False:
      # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
      cliParams = {'InputVolume': inputVolume.GetID(), 'OutputVolume': outputVolume.GetID(), 'ThresholdValue' : imageThreshold, 'ThresholdType' : 'Above'}
      cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    logging.info('Processing completed')

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
