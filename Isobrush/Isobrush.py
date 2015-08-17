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

    if self.developerMode:
      self.reload = qt.QPushButton("Reload", self.frame)
      self.reload.objectName = self.__class__.__name__ + 'Apply'
      self.reload.setToolTip("Reload this effect")
      self.frame.layout().addWidget(self.reload)
      self.widgets.append(self.reload)
      self.connections.append( (self.reload, 'clicked()', self.onReload) )

    HelpButton(self.frame, "As of yet, this is a sample with no real functionality.")

    self.connections.append( (self.apply, 'clicked()', self.onApply) )

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
    super(IsobrushEffectTool,self).__init__(sliceWidget)
    # create a logic instance to do the non-gui work
    self.logic = IsobrushEffectLogic(self.sliceWidget.sliceLogic())

    # interaction state variables
    self.actionState = None

    # initialization
    self.xyPoints = vtk.vtkPoints()
    self.rasPoints = vtk.vtkPoints()
    self.polyData = self.createPolyData()

    self.mapper = vtk.vtkPolyDataMapper2D()
    self.actor = vtk.vtkActor2D()
    self.mapper.SetInputData(self.polyData)
    self.actor.SetMapper(self.mapper)
    property_ = self.actor.GetProperty()
    property_.SetColor(1,1,0)
    property_.SetLineWidth(1)
    self.renderer.AddActor2D( self.actor )
    self.actors.append( self.actor )

    self.initialized = True

  def cleanup(self):
    super(IsobrushEffectTool,self).cleanup()

  def processEvent(self, caller=None, event=None):
    """
    handle events from the render window interactor
    """

    # let the superclass deal with the event if it wants to
    if super(IsobrushEffectTool,self).processEvent(caller,event):
      return

    if event == "LeftButtonPressEvent":
      self.actionState = "painting"
      self.abortEvent(event)
    if event == "MouseMoveEvent":
      xy = self.interactor.GetEventPosition()
      if self.actionState == "painting":
        self.updateIsocurve(xy)
      else:
        self.createIsocurve(xy)
    if event == "LeftButtonReleaseEvent":
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

    self.positionActors()

  def createIsocurve(self, xy):
    print('create', xy)
    sliceLogic = self.sliceWidget.sliceLogic()
    backgroundLogic = sliceLogic.GetBackgroundLayer()
    print(backgroundLogic.GetReslice().GetOutput().GetDimensions())

    # TODO: generate a radiating ring of points in xy space that have
    # similar intensities to the value at xy (use addPoint after 
    # mapping them to RAS).  Need a radius in xy space to search in (maybe
    # re-use paint radius? or make it a fraction of Dimensions?)

  def updateIsocurve(self, xy):
    print('update', xy)
    # TODO: look at all the current points and if any are closer than radius
    # look at moving them out, depending on same criteria use to create them.
    # Afterwards do a pass and if there are any points that are too far apart
    # split them and then re-radiate them from xy.
    # Splitting step may require making a copy of the points, resetting,
    # and adding new points.

  def applyIsocurve(self):
    print('apply')
    # TODO: should be normal apply operation using ras points

  def positionActors(self):
    """
    update draw feedback to follow slice node
    """
    if not hasattr(self, "xyPoints"):
      # ignore events during initialization
      return
    sliceLogic = self.sliceWidget.sliceLogic()
    sliceNode = sliceLogic.GetSliceNode()
    rasToXY = vtk.vtkTransform()
    rasToXY.SetMatrix( sliceNode.GetXYToRAS() )
    rasToXY.Inverse()
    self.xyPoints.Reset()
    rasToXY.TransformPoints( self.rasPoints, self.xyPoints )
    self.polyData.Modified()
    self.sliceView.scheduleRender()

  def createPolyData(self):
    """make an empty single-polyline polydata"""

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(self.xyPoints)

    lines = vtk.vtkCellArray()
    polyData.SetLines(lines)
    idArray = lines.GetData()
    idArray.Reset()
    idArray.InsertNextTuple1(0)

    polygons = vtk.vtkCellArray()
    polyData.SetPolys(polygons)
    idArray = polygons.GetData()
    idArray.Reset()
    idArray.InsertNextTuple1(0)

    return polyData

  def resetPolyData(self):
    """return the polyline to initial state with no points"""
    lines = self.polyData.GetLines()
    idArray = lines.GetData()
    idArray.Reset()
    idArray.InsertNextTuple1(0)
    self.xyPoints.Reset()
    self.rasPoints.Reset()
    lines.SetNumberOfCells(0)
    self.activeSlice = None

  def addPoint(self,ras):
    """add a world space point to the current outline"""
    # store active slice when first point is added
    sliceLogic = self.sliceWidget.sliceLogic()
    currentSlice = sliceLogic.GetSliceOffset()
    if not self.activeSlice:
      self.activeSlice = currentSlice
      self.setLineMode("solid")

    # don't allow adding points on except on the active slice (where
    # first point was laid down)
    if self.activeSlice != currentSlice: return

    # keep track of node state (in case of pan/zoom)
    sliceNode = sliceLogic.GetSliceNode()
    self.lastInsertSliceNodeMTime = sliceNode.GetMTime()

    p = self.rasPoints.InsertNextPoint(ras)
    lines = self.polyData.GetLines()
    idArray = lines.GetData()
    idArray.InsertNextTuple1(p)
    idArray.SetTuple1(0, idArray.GetNumberOfTuples()-1)
    lines.SetNumberOfCells(1)


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

