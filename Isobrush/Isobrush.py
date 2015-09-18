import os
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

import EditorLib
from EditorLib.EditOptions import HelpButton
from EditorLib.EditOptions import EditOptions
from EditorLib import EditUtil
from EditorLib import LabelEffect

#
# IsobrushEffectOptions - see LabelEffect, EditOptions and Effect for superclasses
#

class IsobrushEffectOptions(EditorLib.LabelEffectOptions):
  """ IsobrushEffect-specfic gui
  """

  def __init__(self, parent=0):
    super(IsobrushEffectOptions,self).__init__(parent)

    # self.attributes should be tuple of options:
    # 'MouseTool' - grabs the cursor
    # 'Nonmodal' - can be applied while another is active
    # 'Disabled' - not available
    self.attributes = ('MouseTool')
    self.displayName = 'IsobrushEffect Effect'
    settings = qt.QSettings()
    self.developerMode = settings.value('Developer/DeveloperMode').lower() == 'true'

  def __del__(self):
    super(IsobrushEffectOptions,self).__del__()

  def create(self):
    super(IsobrushEffectOptions,self).create()
    self.apply = qt.QPushButton("Apply", self.frame)
    self.apply.objectName = self.__class__.__name__ + 'Apply'
    self.apply.setToolTip("Apply the extension operation")
    self.frame.layout().addWidget(self.apply)
    self.widgets.append(self.apply)
    self.connections.append( (self.apply, 'clicked()', self.onApply) )

    if self.developerMode:
      self.reload = qt.QPushButton("Reload", self.frame)
      self.reload.objectName = self.__class__.__name__ + 'Apply'
      self.reload.setToolTip("Reload this effect")
      self.frame.layout().addWidget(self.reload)
      self.widgets.append(self.reload)
      self.connections.append( (self.reload, 'clicked()', self.onReload) )

    HelpButton(self.frame, "This is a fancy paint brush.")

    # Add vertical spacer
    self.frame.layout().addStretch(1)

  def destroy(self):
    super(IsobrushEffectOptions,self).destroy()

  # note: this method needs to be implemented exactly as-is
  # in each leaf subclass so that "self" in the observer
  # is of the correct type
  def updateParameterNode(self, caller, event):
    node = EditUtil.EditUtil().getParameterNode()
    if node != self.parameterNode:
      if self.parameterNode:
        node.RemoveObserver(self.parameterNodeTag)
      self.parameterNode = node
      self.parameterNodeTag = node.AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateGUIFromMRML)

  def setMRMLDefaults(self):
    super(IsobrushEffectOptions,self).setMRMLDefaults()

  def updateGUIFromMRML(self,caller,event):
    self.disconnectWidgets()
    super(IsobrushEffectOptions,self).updateGUIFromMRML(caller,event)
    self.connectWidgets()

  def onApply(self):
    print('No really, this is just an example - nothing here yet, but things are getting better')

  def onReload(self):
    EditUtil.EditUtil().setCurrentEffect("DefaultTool")
    import Isobrush
    # TODO: this causes a flash of the new widget, but not a big deal since it's only devel mode
    w = Isobrush.IsobrushWidget()
    w.onReload()
    del(w.parent)
    EditUtil.EditUtil().setCurrentEffect("Isobrush")

  def updateMRMLFromGUI(self):
    if self.updatingGUI:
      return
    disableState = self.parameterNode.GetDisableModifiedEvent()
    self.parameterNode.SetDisableModifiedEvent(1)
    super(IsobrushEffectOptions,self).updateMRMLFromGUI()
    self.parameterNode.SetDisableModifiedEvent(disableState)
    if not disableState:
      self.parameterNode.InvokePendingModifiedEvent()


#
# IsobrushEffectTool
#

class IsobrushEffectTool(LabelEffect.LabelEffectTool):
  """
  One instance of this will be created per-view when the effect
  is selected.  It is responsible for implementing feedback and
  label map changes in response to user input.
  This class observes the editor parameter node to configure itself
  and queries the current view for background and label volume
  nodes to operate on.
  """

  def __init__(self, sliceWidget):
    self.initialized = False
    super(IsobrushEffectTool,self).__init__(sliceWidget)
    # create a logic instance to do the non-gui work
    self.logic = IsobrushEffectLogic(self.sliceWidget.sliceLogic())

    # interaction state variables - track if painting or not
    self.actionState = None

    #
    # cursor actor (paint preview)
    #
    self.cursorMapper = vtk.vtkImageMapper()
    self.cursorDummyImage = vtk.vtkImageData()
    self.cursorDummyImage.AllocateScalars(vtk.VTK_UNSIGNED_INT, 1)
    self.cursorMapper.SetInputData( self.cursorDummyImage )
    self.cursorActor = vtk.vtkActor2D()
    self.cursorActor.VisibilityOff()
    self.cursorActor.SetMapper( self.cursorMapper )
    self.cursorMapper.SetColorWindow( 255 )
    self.cursorMapper.SetColorLevel( 128 )

    self.actors.append( self.cursorActor )

    self.renderer.AddActor2D( self.cursorActor )

    #
    # Shader computation
    # - need to import class from module here since it may not be in sys.path
    #   at startup time
    # - uses dummy render window for framebuffer object context
    #
    from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLShaderComputation
    self.shaderComputation=vtkOpenGLShaderComputation()
    from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLTextureImage
    self.backgroundTextureImage=vtkOpenGLTextureImage()
    self.labelTextureImage=vtkOpenGLTextureImage()
    self.resultImageTexture=vtkOpenGLTextureImage()
    self.iterationImageTexture=vtkOpenGLTextureImage()
    self.backgroundTextureImage.SetShaderComputation(self.shaderComputation)
    self.labelTextureImage.SetShaderComputation(self.shaderComputation)
    self.resultImageTexture.SetShaderComputation(self.shaderComputation)
    self.iterationImageTexture.SetShaderComputation(self.shaderComputation)

    self.shaderComputation.SetVertexShaderSource("""
      #version 120
      attribute vec3 vertexAttribute;
      attribute vec2 textureCoordinateAttribute;
      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        interpolatedTextureCoordinate = vec3(textureCoordinateAttribute, .5);
        gl_Position = vec4(vertexAttribute, 1.);
      }
    """)

    self.initialized = True

    self.previewOn()

  def cleanup(self):
    super(IsobrushEffectTool,self).cleanup()

  def processEvent(self, caller=None, event=None):
    """
    handle events from the render window interactor
    """
    if not self.initialized:
      return

    # let the superclass deal with the event if it wants to
    try:
      if super(IsobrushEffectTool,self).processEvent(caller,event):
        return
    except TypeError:
      # this can happen when an event comes in during destruction of the object
      pass

    if event == "LeftButtonPressEvent":
      self.actionState = "painting"
      self.abortEvent(event)
    if event == "MouseMoveEvent":
      xy = self.interactor.GetEventPosition()
      self.previewOn(xy)
      if self.actionState == "painting":
        self.updateIsocurve(xy)
      else:
        self.createIsocurve(xy)
    if event == "LeftButtonReleaseEvent":
      self.previewOff()
      self.applyIsocurve()
      self.actionState = None
      self.abortEvent(event)
    else:
      pass

    # events from the slice node
    if caller and caller.IsA('vtkMRMLSliceNode'):
      # here you can respond to pan/zoom or other changes
      # to the view
      pass
      self.previewOn()

  def previewOn(self, xy=(100,100)):

    if not self.editUtil.getBackgroundImage() or not self.editUtil.getLabelImage():
      return

    #
    # get the visible section of the background (pre-window/level)
    # to use as input to the shader code
    #
    sliceLogic = self.sliceWidget.sliceLogic()
    backgroundLogic = sliceLogic.GetBackgroundLayer()
    backgroundLogic.GetReslice().Update()
    backgroundImage = backgroundLogic.GetReslice().GetOutputDataObject(0)
    backgroundDimensions = backgroundImage.GetDimensions()
    self.backgroundTextureImage.SetImageData(backgroundImage)
    self.backgroundTextureImage.Activate(0)
    labelLogic = sliceLogic.GetLabelLayer()
    labelLogic.GetReslice().Update()
    labelImage = labelLogic.GetReslice().GetOutputDataObject(0)
    labelDimensions = labelImage.GetDimensions()
    self.labelTextureImage.SetImageData(labelImage)
    self.labelTextureImage.Activate(1)

    # make a result image to match dimensions and type
    resultImage = vtk.vtkImageData()
    resultImage.SetDimensions(backgroundDimensions)
    resultImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    self.shaderComputation.SetResultImageData(resultImage)
    self.shaderComputation.AcquireResultRenderbuffer()
    self.resultImageTexture.SetImageData(resultImage)
    self.resultImageTexture.Activate(2)

    # make another  result image for iteration
    iterationImage = vtk.vtkImageData()
    iterationImage.SetDimensions(backgroundDimensions)
    iterationImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    self.iterationImageTexture.SetImageData(iterationImage)
    self.iterationImageTexture.Activate(3)

    fragmentShaderSource = """
      #version 120
      varying vec3 interpolatedTextureCoordinate;
      uniform sampler3D textureUnit0; // background
      uniform sampler3D textureUnit1; // label
      uniform sampler3D %(iterationTextureUnit)s; // where to get previous iteration image
      void main()
      {
        vec3 referenceTextureCoordinate = vec3(%(referenceX)f, %(referenceY)f, .5);
        vec3 samplePoint = interpolatedTextureCoordinate;
        // background samples
        vec4 referenceSample = %(sampleScale)f * texture3D(textureUnit0, referenceTextureCoordinate);
        vec4 volumeSample = %(sampleScale)f * texture3D(textureUnit0, interpolatedTextureCoordinate);
        // previous result sample
        vec4 previousSample = %(sampleScale)f * texture3D(%(iterationTextureUnit)s, interpolatedTextureCoordinate);

        gl_FragColor = vec4(0.);

        float brushDistance = distance(vec2(0.,0.), vec2( %(radiusX)f, %(radiusY)f));
        float pixelDistance = distance(referenceTextureCoordinate, interpolatedTextureCoordinate);

        // if the current pixel is in the reference point, always paint it
        if (pixelDistance < brushDistance) {
          gl_FragColor = vec4(1., 1., 0., .5);
        }

        // if the current pixel is in the overall radius
        // and the intensity matches
        // and a neighbor is non-zero then set it
        if (pixelDistance < %(radius)f) {
          if (abs(referenceSample.r - volumeSample.r) < %(similarityThreshold)f) {
            vec3 neighbor;
            neighbor = interpolatedTextureCoordinate + vec3(-brushDistance, 0., 0.);
            if (texture3D(%(iterationTextureUnit)s, neighbor).r > 0.) {
              gl_FragColor = vec4(1., 1., 0., .5);
            }
            neighbor = interpolatedTextureCoordinate + vec3( brushDistance, 0., 0.);
            if (texture3D(%(iterationTextureUnit)s, neighbor).r > 0.) {
              gl_FragColor = vec4(1., 1., 0., .5);
            }
            neighbor = interpolatedTextureCoordinate + vec3( 0, -brushDistance, 0.);
            if (texture3D(%(iterationTextureUnit)s, neighbor).r > 0.) {
              gl_FragColor = vec4(1., 1., 0., .5);
            }
            neighbor = interpolatedTextureCoordinate + vec3( 0,  brushDistance, 0.);
            if (texture3D(%(iterationTextureUnit)s, neighbor).r > 0.) {
              gl_FragColor = vec4(1., 1., 0., .5);
            }
          }
        }
      }
    """ % {
      'sampleScale' : 500.,
      'similarityThreshold' : 0.1,
      'referenceX' : xy[0] / float(backgroundDimensions[0]),
      'referenceY' : xy[1] / float(backgroundDimensions[1]),
      'radiusX' : 1. / backgroundDimensions[0],
      'radiusY' : 1. / backgroundDimensions[1],
      'radius'     : 0.5,
      'iterationTextureUnit'     : "%(iterationTextureUnit)s",
    }

    for iteration in range(99):
      if iteration % 2:
        self.iterationImageTexture.AttachAsDrawTarget(0, 0, 0)
        iterationTextureUnit = "textureUnit2"
      else:
        self.resultImageTexture.AttachAsDrawTarget(0, 0, 0)
        iterationTextureUnit = "textureUnit3"
      self.shaderComputation.SetFragmentShaderSource(fragmentShaderSource % {
        'iterationTextureUnit' : iterationTextureUnit
      })
      self.shaderComputation.Compute()

    self.shaderComputation.AcquireResultRenderbuffer()
    self.shaderComputation.SetFragmentShaderSource(fragmentShaderSource % {
      'iterationTextureUnit' : "textureUnit3"
    })
    self.shaderComputation.Compute()
    self.shaderComputation.ReadResult()
    self.shaderComputation.ReleaseResultRenderbuffer()

    self.cursorMapper.SetInputDataObject(resultImage)
    self.cursorActor.VisibilityOn()
    self.sliceView.forceRender()

  def previewOff(self):
    self.cursorActor.VisibilityOff()
    self.sliceView.scheduleRender()

  def createIsocurve(self, xy):
    # print('create', xy)
    sliceLogic = self.sliceWidget.sliceLogic()
    backgroundLogic = sliceLogic.GetBackgroundLayer()
    # print(backgroundLogic.GetReslice().GetOutput().GetDimensions())

  def updateIsocurve(self, xy):
    # print('update', xy)
    pass

  def applyIsocurve(self):
    print('apply')
    # TODO: should be normal apply operation using ras points


#
# IsobrushEffectLogic
#

class IsobrushEffectLogic(LabelEffect.LabelEffectLogic):
  """
  This class contains helper methods for a given effect
  type.  It can be instanced as needed by an IsobrushEffectTool
  or IsobrushEffectOptions instance in order to compute intermediate
  results (say, for user feedback) or to implement the final
  segmentation editing operation.  This class is split
  from the IsobrushEffectTool so that the operations can be used
  by other code without the need for a view context.
  """

  def __init__(self,sliceLogic):
    self.sliceLogic = sliceLogic

  def apply(self,xy):
    pass

#
# The IsobrushEffect class definition
#

class IsobrushEffectExtension(LabelEffect.LabelEffect):
  """Organizes the Options, Tool, and Logic classes into a single instance
  that can be managed by the EditBox
  """

  def __init__(self):
    # name is used to define the name of the icon image resource (e.g. IsobrushEffect.png)
    self.name = "IsobrushEffect"
    # tool tip is displayed on mouse hover
    self.toolTip = "Isobrush: adaptive paint brush using image intensity to define shape"

    self.options = IsobrushEffectOptions
    self.tool = IsobrushEffectTool
    self.logic = IsobrushEffectLogic
#
# Isobrush
#

class Isobrush(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Isobrush" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Segmentation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Steve Pieper (Isomics, Inc.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    This is an editor effect plugin that uses a custom shader.
    """
    self.parent.acknowledgementText = """
    This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
    and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

    # Add this extension to the editor's list for discovery when the module
    # is created.  Since this module may be discovered before the Editor itself,
    # create the list if it doesn't already exist.
    try:
      slicer.modules.editorExtensions
    except AttributeError:
      slicer.modules.editorExtensions = {}
    slicer.modules.editorExtensions['Isobrush'] = IsobrushEffectExtension

#
# IsobrushWidget
#

class IsobrushWidget(ScriptedLoadableModuleWidget):
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


    # Add vertical spacer
    self.layout.addStretch(1)


  def cleanup(self):
    pass

  def onReload(self):
    """
    Run the regular reload and then
    register the update extension with the Editor module
    """
    ScriptedLoadableModuleWidget.onReload(self)

    slicer.modules.editorExtensions['Isobrush'] = IsobrushEffectExtension


#
# IsobrushLogic
#

class IsobrushLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """


class IsobrushTest(ScriptedLoadableModuleTest):
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
    self.test_Isobrush1()

  def test_Isobrush1(self):
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
    logic = IsobrushLogic()
    self.assertTrue( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')

