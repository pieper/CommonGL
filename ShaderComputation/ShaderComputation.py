import os
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

#
# ShaderComputation
#

class ShaderComputation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ShaderComputation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
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
# ShaderComputationWidget
#

class ShaderComputationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    self.amplitude = 0.
    self.frequency = 0.
    self.phase = 0.

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
    # render volume selector
    #
    self.renderSelector = slicer.qMRMLNodeComboBox()
    self.renderSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.renderSelector.selectNodeUponCreation = True
    self.renderSelector.addEnabled = False
    self.renderSelector.removeEnabled = False
    self.renderSelector.noneEnabled = False
    self.renderSelector.showHidden = False
    self.renderSelector.showChildNodeTypes = True
    self.renderSelector.setMRMLScene( slicer.mrmlScene )
    self.renderSelector.setToolTip( "Pick volume to render." )
    parametersFormLayout.addRow("Render Volume: ", self.renderSelector)

    #
    # amplitude value
    #
    self.transformAmplitude = ctk.ctkSliderWidget()
    self.transformAmplitude.singleStep = 0.001
    self.transformAmplitude.minimum = -2
    self.transformAmplitude.maximum = 2
    self.transformAmplitude.value = 0.
    self.transformAmplitude.setToolTip("Amount of the transform to apply")
    parametersFormLayout.addRow("Transform Amplitude", self.transformAmplitude)

    #
    # frequency value
    #
    self.transformFrequency = ctk.ctkSliderWidget()
    self.transformFrequency.singleStep = 0.0001
    self.transformFrequency.minimum = -.5
    self.transformFrequency.maximum = .5
    self.transformFrequency.value = 0.1
    self.transformFrequency.setToolTip("Frequency the transform")
    parametersFormLayout.addRow("Transform Frequency", self.transformFrequency)

    #
    # phase value
    #
    self.transformPhase = ctk.ctkSliderWidget()
    self.transformPhase.singleStep = 0.001
    self.transformPhase.minimum = -5
    self.transformPhase.maximum = 5
    self.transformPhase.value = 0.1
    self.transformPhase.setToolTip("Phase the transform")
    parametersFormLayout.addRow("Transform Phase", self.transformPhase)

    #
    # check box to apply transform
    #
    self.applyTransformCheckBox = qt.QCheckBox()
    self.applyTransformCheckBox.checked = True
    self.applyTransformCheckBox.setToolTip("If checked, render with transform applied.")
    parametersFormLayout.addRow("Apply Transform", self.applyTransformCheckBox)

    # connections
    self.renderSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onVolumeChange)
    self.transformAmplitude.connect("valueChanged(double)", self.onTransformChange)
    self.transformFrequency.connect("valueChanged(double)", self.onTransformChange)
    self.transformPhase.connect("valueChanged(double)", self.onTransformChange)
    self.applyTransformCheckBox.connect("toggled(bool)", self.onTransformChange)

    # Add vertical spacer
    self.layout.addStretch(1)

    # set up the scene renderer
    self.sceneRenderer = SceneRenderer()
    self.sceneRenderer.transformPointSource = self.demoTransformPointSource()

  def cleanup(self):
    self.sceneRenderer.cleanup()

  def demoTransformPointSource(self):
    return ("""
      vec3 transformPoint(const in vec3 samplePoint)
      // Apply a spatial transformation to a world space point
      {
          // TODO: get MRMLTransformNodes as vector fields
          float frequency = %(frequency)f;
          float phase = %(phase)f;
          return samplePoint + %(amplitude)f * vec3(samplePoint.x * sin(phase + frequency * samplePoint.z),
                                                    samplePoint.y * cos(phase + frequency * samplePoint.z),
                                                    0);
      }
    """ % {
        'amplitude' : self.amplitude,
        'frequency' : self.frequency,
        'phase' : self.phase
    })

  def onTransformChange(self):
    """Perform the render when any input changes"""
    self.amplitude = 0.
    self.frequency = 0.
    self.phase = 0.
    if self.applyTransformCheckBox.checked:
      self.amplitude = self.transformAmplitude.value
      self.frequency = self.transformFrequency.value
      self.phase = self.transformPhase.value
    self.sceneRenderer.transformPointSource = self.demoTransformPointSource()
    self.sceneRenderer.render()

  def onVolumeChange(self):
    """Perform the render when any input changes"""
    self.sceneRenderer.setVolume(self.renderSelector.currentNode())
    self.sceneRenderer.render()


from slicer.util import VTKObservationMixin
class SceneObserver(VTKObservationMixin):
  """Observes everything in the scene
  TODO: add a simple way for users of this class to subscribe
  to patterns of events, such as adds/removals of certain classes
  of nodes, or events from any node of a given node class.
  TODO: if this experiment proves useful it could migrate to slicer.util
  """

  def __init__(self, scene=None):
    """Add observers to the mrmlScene and also to all the nodes of the scene"""
    VTKObservationMixin.__init__(self)

    if scene:
      self._scene = scene
    else:
      self._scene = slicer.mrmlScene
    self.addObserver(scene, scene.NodeAddedEvent, self.onNodeAdded)
    self.addObserver(scene, scene.NodeRemovedEvent, self.onNodeRemoved)

    scene.InitTraversal()
    node = scene.GetNextNode()
    while node:
      self._observeNode(node)
      node = scene.GetNextNode()

    # a dictionary of (nodeKey,eventKey) -> callback
    # where nodeKey is classname "vtkMRML{nodeKey}Node"
    # and eventKey is tested for equality to "{eventKey}Event"
    self._triggers = {}

    # for debugging turn this on
    # 0: no debugging
    # 1: print each trigger
    # 2: print each scene event
    self.verbose = 0

  def __del__(self):
    self.removeObservers()

  def _observeNode(self,node):
    if node.IsA('vtkMRMLNode'):
      # use AnyEvent since it will catch events like TransformModified
      self.addObserver(node, vtk.vtkCommand.AnyEvent, self._onNodeModified)
    else:
      raise('should not happen: non node is in scene')

  @vtk.calldata_type(vtk.VTK_OBJECT)
  def onNodeAdded(self, caller, event, calldata):
    node = calldata
    if not self.hasObserver(node, vtk.vtkCommand.AnyEvent, self._onNodeModified):
      self._observeNode(node)
    self._trigger(node.GetClassName(), "NodeAddedEvent")

  @vtk.calldata_type(vtk.VTK_OBJECT)
  def onNodeRemoved(self, caller, event, calldata):
    node = calldata
    self.removeObserver(node, vtk.vtkCommand.AnyEvent, self._onNodeModified)
    self._trigger(node.GetClassName(), "NodeRemovedEvent")

  def _trigger(self, className, eventName):
    key = (className,eventName)
    if key in self._triggers:
      for callback in self._triggers[key]:
        if self.verbose > 0:
          print("callback triggered by %s from %s" % (eventName, className))
        callback()

  def _onNodeModified(self,node,eventName):
    if self.verbose > 1:
      print("%s from %s" % (eventName, node.GetID()))
    self._trigger(node.GetClassName(), eventName)

  def _key(self, nodeKey, eventKey):
    """Convert shorthand notation to full class and event name"""
    return ("vtkMRML%sNode"%nodeKey,"%sEvent"%eventKey)

  def addTrigger(self, nodeKey, eventKey, callback):
    key = self._key(nodeKey, eventKey)
    if not key in self._triggers:
      self._triggers[key] = []
    self._triggers[key].append(callback)

  def removeTrigger(self, nodeKey, eventKey, callback):
    key = self._key(nodeKey, eventKey)
    if key in self._triggers:
      if callback in self._triggers[key]:
        self._triggers[key].remove(callback)


class FieldSampler(object):
  """This is a generic superclass for objects that define samplable fields
  in space based on data in the mrml scene.  Being samplable means that the
  class can generate an rgba and spatial gradient at any point in space.
  (TODO: spacetime).
  A field sampler is associated with a shaderComputation instance which
  defines the rendering context and a textureUnit that is available
  to store a vtkImageData.
  The field sampler includes methods that return glsl code
  needed to implement the sampling operation.
  """

  def __init__(self, shaderComputation, textureUnit, node):
    self.shaderComputation = shaderComputation
    self.textureUnit = textureUnit
    self.node = node

  def checkResources(self):
    """TODO: look at the data and determine how much of the available OpenGL
    resources will be needed to represent it.  Intended to be overridden"""
    logging.warn("FieldSampler.checkResources method should be overriden by subclass")

  def updateFromMRML(self):
    """Do whatever is needed to pass data or state from MRML
    nodes into GPU context.  Intended to be overridden"""
    logging.warn("FieldSampler.updateFromMRML method should be overriden by subclass")

  def fieldSampleSource(self):
    """Return a snippet of glsl code that can be used to
    sample this field.  Intended to be overridden"""
    logging.warn("FieldSampler.fieldSampleSource method should be overriden by subclass")



class Fiducials(FieldSampler):
  """Treats a fiducial point as a spherical field.
  """

  def __init__(self, shaderComputation, textureUnit, node):
    self.shaderComputation = shaderComputation
    self.textureUnit = textureUnit
    self.node = node

    self.dummyImage = vtk.vtkImageData()
    self.dummyImage.SetDimensions(1,1,1)
    self.dummyImage.AllocateScalars(vtk.VTK_SHORT, 1)
    try:
      from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLTextureImage
    except ImportError:
      import vtkAddon
      vtkOpenGLTextureImage=vtkAddon.vtkOpenGLTextureImage
    self.textureImage=vtkOpenGLTextureImage()
    self.textureImage.SetShaderComputation(self.shaderComputation)
    self.textureImage.SetImageData(self.dummyImage)

  def updateFromMRML(self):
    """Do whatever is needed to pass data or state from MRML
    nodes into GPU context.  Intended to be overridden"""
    pass
    self.textureImage.Activate(self.textureUnit);

  def transferFunctionSource(self):
    """Create source code for transfer function that maps
    a sample and gradient to a color and opacity based on
    the passed volumePropertyNode.
    """
    source = """
    void transferFunction%(textureUnit)s(const in float sample, const in float gradientMagnitude,
                          out vec3 color, out float opacity)
    {
       color = vec3(1,0,0);
       opacity = sample;
    }
    """ % { 'textureUnit' : self.textureUnit }
    return source

  def fieldSampleSource(self):
    """Return the GLSL code to sample our volume in space"""

    transferFunctionSource = self.transferFunctionSource()

    shaderFiducials = self.node

    # each fiducial is checked and if inside, returns sample
    fiducialSampleTemplate = """
        centerToSample = samplePoint-vec3( %(fiducialString)s );
        distance = length(centerToSample);
        if (distance < glow * %(radius)s) {
            sample += smoothstep(distance / glow, distance * glow, distance);
            normal += normalize(centerToSample);
        }
    """

    fiducialDisplayNode = shaderFiducials.GetDisplayNode()
    radius = fiducialDisplayNode.GetGlyphScale()
    fiducialCenter = [0,]*3
    fiducialSampleSource = ""
    for markupIndex in range(shaderFiducials.GetNumberOfMarkups()):
      shaderFiducials.GetNthFiducialPosition(markupIndex,fiducialCenter)
      fiducialCenterString = "%f,%f,%f" % tuple(fiducialCenter)
      fiducialSampleSource += fiducialSampleTemplate % {
                'radius' : str(radius),
                'fiducialString' : fiducialCenterString
              }


    fieldSampleSource = """
      void sampleVolume%(textureUnit)s(const in sampler3D volumeTextureUnit, const in vec3 samplePointIn, const in float gradientSize,
                        out float sample, out vec3 normal, out float gradientMagnitude)
      {
        // TODO: transform should be associated with the sampling, not the ray point
        //       so that gradient is calculated incorporating transform
        vec3 samplePoint = transformPoint(samplePointIn);

        vec3 centerToSample;
        float distance;
        float glow = 1.2;

        // default if sample is not in fiducial
        sample = 0.;
        normal = vec3(0,0,0);
        gradientMagnitude = 1.;

        %(fiducialSampleSource)s

        normal = normalize(normal);

      }
    """ % {
        'fiducialSampleSource' : fiducialSampleSource,
        'textureUnit' : self.textureUnit,
        }
    return transferFunctionSource + fieldSampleSource


class VolumeTexture(FieldSampler):
  """Maps a volume node to a GLSL renderable collection
  of textures and code"""

  def __init__(self, shaderComputation, textureUnit, volumeNode, optimizeDynamicRange=True):
    FieldSampler.__init__(self, shaderComputation, textureUnit, volumeNode)
    self.shiftScale = vtk.vtkImageShiftScale()
    self.optimizeDynamicRange = optimizeDynamicRange

    try:
      from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLTextureImage
    except ImportError:
      import vtkAddon
      vtkOpenGLTextureImage=vtkAddon.vtkOpenGLTextureImage
    self.textureImage=vtkOpenGLTextureImage()
    self.textureImage.SetShaderComputation(self.shaderComputation)

    self.updateFromMRML()

  def updateFromMRML(self):
    if self.optimizeDynamicRange:
      # since the OpenGL texture will be floats in the range 0 to 1,
      # all negative values will get clamped to zero.
      # Also if the sample values aren't evenly spread through
      # the zero-to-one space we may run into numerical issues.
      # So rescale the data to the
      # to fit in the full range of the a 16 bit short.
      # (any vtkImageData scalar type should work with this approach)
      self.shiftScale.SetInputData(self.node.GetImageData())
      self.shiftScale.SetOutputScalarTypeToUnsignedShort()
      low, high = self.node.GetImageData().GetScalarRange()
      self.shiftScale.SetShift(-low)
      if high == low:
        scale = 1.
      else:
        scale = (1. * vtk.VTK_UNSIGNED_SHORT_MAX) / (high-low)
      self.shiftScale.SetScale(scale)
      self.sampleUnshift = low
      self.sampleUnscale = high-low

      self.shiftScale.Update()
      self.textureImage.SetImageData(self.shiftScale.GetOutputDataObject(0))
    else:
      self.sampleUnshift = 0
      self.sampleUnscale = 1
      self.textureImage.SetImageData(self.node.GetImageData())
    self.textureImage.Activate(self.textureUnit)


  def sampleVolumeParameters(self):
    """Calculate the dictionary of substitutions for the
    current state of the volume node in a form for
    substitution into the sampleVolume shader function
    TODO: this would probably be better as uniforms, but that
    requires doing a lot of parsing and data management in C++
    """

    rasToIJK = vtk.vtkMatrix4x4()
    self.node.GetRASToIJKMatrix(rasToIJK)

    transformNode = self.node.GetParentTransformNode()
    if transformNode:
      if transformNode.IsTransformToWorldLinear():
        rasToRAS = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToWorld(rasToRAS)
        rasToRAS.Invert()
        rasToRAS.Multiply4x4(rasToIJK, rasToRAS, rasToIJK)
      else:
        error.warn('Cannot (yet) handle nonlinear transforms')

    # dimensions are number of pixels in (row, column, slice)
    # which maps to 0-1 space of S, T, P
    dimensions = self.node.GetImageData().GetDimensions()
    ijkToSTP = vtk.vtkMatrix4x4()
    ijkToSTP.Identity()
    for diagonal in range(3):
      ijkToSTP.SetElement(diagonal,diagonal, 1./dimensions[diagonal])
    rasToSTP = vtk.vtkMatrix4x4()
    ijkToSTP.Multiply4x4(ijkToSTP, rasToIJK, rasToSTP)

    parameters = {}
    rows = ('rasToS', 'rasToT', 'rasToP')
    for row in range(3):
      rowKey = rows[row]
      parameters[rowKey] = ""
      for col in range(4):
        element = rasToSTP.GetElement(row,col)
        parameters[rowKey] += "%f," % element
      parameters[rowKey] = parameters[rowKey][:-1] # clear trailing comma

    # since texture is 0-1, take into account both pixel spacing
    # and dimension as layed out in memory so that the normals
    # is calculated in a uniform space
    spacings = self.node.GetSpacing()
    parameters['mmToS'] = spacings[0] / dimensions[0]
    parameters['mmToT'] = spacings[1] / dimensions[1]
    parameters['mmToP'] = spacings[2] / dimensions[2]

    # the inverse transpose of the upper 3x3 of the stpToRAS matrix,
    # which is the transpose of the upper 3x3 of the rasTSTP matrix
    normalSTPToRAS = vtk.vtkMatrix3x3();
    for row in range(3):
      for column in range(3):
        normalSTPToRAS.SetElement(row,column, rasToSTP.GetElement(row,column));
    normalSTPToRAS.Transpose()
    parameters['normalSTPToRAS'] = ''
    for column in range(3):
      for row in range(3):
        # write in column-major order for glsl mat3 constructor
        parameters['normalSTPToRAS'] += "%f," % normalSTPToRAS.GetElement(row,column)
    parameters['normalSTPToRAS'] = parameters['normalSTPToRAS'][:-1] # clear trailing comma

    return parameters

  def transferFunctionSource(self):
    """Create source code for transfer function that maps
    a sample and gradient to a color and opacity based on
    the passed volumePropertyNode.
    """

    displayNode = ShaderComputationLogic().volumeRenderingDisplayNode(self.node)
    volumePropertyNode = displayNode.GetVolumePropertyNode()
    scalarOpacityFunction = volumePropertyNode.GetScalarOpacity(0)
    gradientOpacityFunction = volumePropertyNode.GetGradientOpacity(0)
    colorTransfer = volumePropertyNode.GetColor(0)

    source = """
    void transferFunction%(textureUnit)s(const in float sample, const in float gradientMagnitude,
                          out vec3 color, out float opacity)
    {
      float scalarOpacity = 0., gradientOpacity = 0.;

    """ % { 'textureUnit' : self.textureUnit }

    # convert the scalarOpacity transfer function to a procedure
    # - ignore the interpolation options; only linear interpolation
    intensities = []
    scalarOpacities = []
    size = scalarOpacityFunction.GetSize()
    values = [0,]*4
    for index in range(size):
      scalarOpacityFunction.GetNodeValue(index, values)
      intensities.append(values[0])
      scalarOpacities.append(values[1])
    source += """
      if (sample < %(minIntensity)f) {
        scalarOpacity = %(minOpacity)f;
      }
    """ % {'minIntensity': intensities[0], 'minOpacity': scalarOpacities[0]}
    for index in range(size-1):
      currentIndex = index + 1
      source += """
        else if (sample < %(currentIntesity)f) {
          scalarOpacity = mix(%(lastOpacity)f, %(currentOpacity)f, (sample - %(lastIntensity)f) / %(intensityRange)f);
        }
      """ % {'currentIntesity': intensities[currentIndex],
             'lastOpacity': scalarOpacities[index],
             'currentOpacity': scalarOpacities[currentIndex],
             'lastIntensity': intensities[index],
             'intensityRange': intensities[currentIndex] - intensities[index],
             }
    source += """
      else {
        scalarOpacity = %(lastOpacity)f;
      }
    """ % {'lastOpacity': scalarOpacities[size-1]}

    # convert the gradientOpacity transfer function to a procedure
    # - ignore the interpolation options; only linear interpolation
    intensities = []
    gradientOpacities = []
    size = gradientOpacityFunction.GetSize()
    values = [0,]*4
    for index in range(size):
      gradientOpacityFunction.GetNodeValue(index, values)
      intensities.append(values[0])
      gradientOpacities.append(values[1])
    source += """
      if (sample < %(minIntensity)f) {
        gradientOpacity = %(minOpacity)f;
      }
    """ % {'minIntensity': intensities[0], 'minOpacity': gradientOpacities[0]}
    for index in range(size-1):
      currentIndex = index + 1
      source += """
        else if (sample < %(currentIntesity)f) {
          gradientOpacity = mix(%(lastOpacity)f, %(currentOpacity)f, (sample - %(lastIntensity)f) / %(intensityRange)f);
        }
      """ % {'currentIntesity': intensities[currentIndex],
             'lastOpacity': gradientOpacities[index],
             'currentOpacity': gradientOpacities[currentIndex],
             'lastIntensity': intensities[index],
             'intensityRange': intensities[currentIndex] - intensities[index],
             }
    source += """
      else {
        gradientOpacity = %(lastOpacity)f;
      }
    """ % {'lastOpacity': gradientOpacities[size-1]}

    source += """
      opacity = scalarOpacity + gradientOpacity;
    """

    # convert the colorTransfer to a procedure
    intensities = []
    colors = []
    size = colorTransfer.GetSize()
    values = [0,]*6
    for index in range(size):
      colorTransfer.GetNodeValue(index, values)
      intensities.append(values[0])
      colors.append("vec3" + str(tuple(values[1:4])))
    source += """
      if (sample < %(minIntensity)f) {
        color = %(minColor)s;
      }
    """ % {'minIntensity': intensities[0], 'minColor': colors[0]}
    for index in range(size-1):
      currentIndex = index + 1
      source += """
        else if (sample < %(currentIntesity)f) {
          color = mix(%(lastColor)s, %(currentColor)s, (sample - %(lastIntensity)f) / %(intensityRange)f);
        }
      """ % {'currentIntesity': intensities[currentIndex],
             'lastColor': colors[index],
             'currentColor': colors[currentIndex],
             'lastIntensity': intensities[index],
             'intensityRange': intensities[currentIndex] - intensities[index],
             }
    source += """
      else {
        color = %(lastColor)s;
      }

    """ % {'lastColor': colors[size-1]}
    source += """
    }
    """
    return source

  def fieldSampleSource(self):
    """Return the GLSL code to sample our volume in space
    Function names are postpended with texure unit number to avoid name clashes.
    """

    displayNode = ShaderComputationLogic().volumeRenderingDisplayNode(self.node)
    if displayNode.GetVisibility() == 0:
      return """
        void transferFunction%(textureUnit)s(
            const in float sample, const in float gradientMagnitude,
            out vec3 color, out float opacity) {
          color = vec3(0);
          opacity = 0;
        }
        void sampleVolume%(textureUnit)s(
            const in sampler3D volumeTextureUnit,
            const in vec3 samplePointIn, const in float gradientSize,
            out float sample, out vec3 normal, out float gradientMagnitude) {
          sample = 0;
          normal = vec3(0,0,0);
          gradientMagnitude = 0;
      }
      """ % { 'textureUnit': self.textureUnit }

    transferFunctionSource = self.transferFunctionSource()

    sampleVolumeParameters = self.sampleVolumeParameters()
    sampleVolumeParameters.update({
          'sampleUnshift' : self.sampleUnshift,
          'sampleUnscale' : self.sampleUnscale,
          'textureUnit' : self.textureUnit,
    })
    fieldSampleSource = """
      float textureSampleDenormalized%(textureUnit)s(const in sampler3D volumeTextureUnit, const in vec3 stpPoint) {
        return ( texture3D(volumeTextureUnit, stpPoint).r * %(sampleUnscale)f + %(sampleUnshift)f );
      }

      void sampleVolume%(textureUnit)s(const in sampler3D volumeTextureUnit, const in vec3 samplePointIn, const in float gradientSize,
                        out float sample, out vec3 normal, out float gradientMagnitude)
      {

        // TODO: transform should be applied to each sample in the gradient estimation
        //       so that gradient is calculated incorporating transform.
        vec3 samplePoint = transformPoint(samplePointIn);

        // vectors to map RAS to stp
        vec4 rasToS =  vec4( %(rasToS)s );
        vec4 rasToT =  vec4( %(rasToT)s );
        vec4 rasToP =  vec4( %(rasToP)s );

        vec3 stpPoint;
        vec4 sampleCoordinate = vec4(samplePoint, 1.);
        stpPoint.s = dot(rasToS,sampleCoordinate);
        stpPoint.t = dot(rasToT,sampleCoordinate);
        stpPoint.p = dot(rasToP,sampleCoordinate);

        if (any(lessThan(stpPoint, vec3(0))) || any(greaterThan(stpPoint,vec3(1)))) {
            sample = 0;
            gradientMagnitude = 0;
            return;
        }

        #define S(point) textureSampleDenormalized%(textureUnit)s(volumeTextureUnit, point)

        // read from 3D texture
        sample = S(stpPoint);

        // central difference sample gradient (P is +1, N is -1)
        float sP00 = S(stpPoint + vec3(%(mmToS)f * gradientSize,0,0));
        float sN00 = S(stpPoint - vec3(%(mmToS)f * gradientSize,0,0));
        float s0P0 = S(stpPoint + vec3(0,%(mmToT)f * gradientSize,0));
        float s0N0 = S(stpPoint - vec3(0,%(mmToT)f * gradientSize,0));
        float s00P = S(stpPoint + vec3(0,0,%(mmToP)f * gradientSize));
        float s00N = S(stpPoint - vec3(0,0,%(mmToP)f * gradientSize));

        #undef S

        // TODO: add Sobel option to filter gradient
        // https://en.wikipedia.org/wiki/Sobel_operator#Extension_to_other_dimensions

        vec3 gradient = vec3( (sP00-sN00),
                              (s0P0-s0N0),
                              (s00P-s00N) );

        gradientMagnitude = length(gradient);

        // https://en.wikipedia.org/wiki/Normal_(geometry)#Transforming_normals
        mat3 normalSTPToRAS = mat3(%(normalSTPToRAS)s);
        vec3 localNormal;
        localNormal = (-1. / gradientMagnitude) * gradient;
        normal = normalize(normalSTPToRAS * localNormal);
      }
    """ % sampleVolumeParameters

    return transferFunctionSource + fieldSampleSource

class LabelMapTexture(VolumeTexture):
  """Most of the functionality is inherited, but a special sampler is used"""

  def __init__(self, shaderComputation, textureUnit, volumeNode):
    FieldSampler.__init__(self, shaderComputation, textureUnit, volumeNode)
    try:
      from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLTextureImage
    except ImportError:
      import vtkAddon
      vtkOpenGLTextureImage=vtkAddon.vtkOpenGLTextureImage
    self.textureImage=vtkOpenGLTextureImage()
    self.textureImage.SetShaderComputation(self.shaderComputation)
    self.textureImage.SetInterpolate(0)

  def transferFunctionSource(self):
    """Create source code for transfer function that maps
    a sample and gradient to a color and opacity based on
    the passed volumePropertyNode.
    """
    # TODO: need to make the label map lookup table
    source = """
    void transferFunction%(textureUnit)s(const in float sample, const in float gradientMagnitude /* TODO: not used */,
                          out vec3 color, out float opacity)
    {
       color = vec3(1,0,0);
       opacity = sample;
    }
    """ % { 'textureUnit' : self.textureUnit }
    return source

  def fieldSampleSource(self):
    """Return the GLSL code to sample our volume in space
    Function names are postpended with texure unit number to avoid name clashes.
    """

    transferFunctionSource = self.transferFunctionSource()

    # TODO:
    # - set neighbors to zero/one depending of they are equal to sample value for gradient
    # - refactor VolumeTexture so we don't repeat here
    # - make a transfer function for color table
    # TODO: need a procedural blend option (cvg-style)
    # TODO: calculate ras bounds for all objects
    sampleVolumeParameters = self.sampleVolumeParameters()
    sampleVolumeParameters.update({
          'sampleUnshift' : self.sampleUnshift,
          'sampleUnscale' : self.sampleUnscale,
          'textureUnit' : self.textureUnit,
    })
    fieldSampleSource = """
      float textureSampleDenormalized%(textureUnit)s(const in sampler3D volumeTextureUnit, const in vec3 stpPoint) {
        return ( texture3D(volumeTextureUnit, stpPoint).r * %(sampleUnscale)f + %(sampleUnshift)f );
      }

      void sampleVolume%(textureUnit)s(const in sampler3D volumeTextureUnit, const in vec3 samplePointIn, const in float gradientSize,
                        out float sample, out vec3 normal, out float gradientMagnitude)
      {

        // TODO: transform should be applied to each sample in the gradient estimation
        //       so that gradient is calculated incorporating transform.
        vec3 samplePoint = transformPoint(samplePointIn);

        // vectors to map RAS to stp
        vec4 rasToS =  vec4( %(rasToS)s );
        vec4 rasToT =  vec4( %(rasToT)s );
        vec4 rasToP =  vec4( %(rasToP)s );

        vec3 stpPoint;
        vec4 sampleCoordinate = vec4(samplePoint, 1.);
        stpPoint.s = dot(rasToS,sampleCoordinate);
        stpPoint.t = dot(rasToT,sampleCoordinate);
        stpPoint.p = dot(rasToP,sampleCoordinate);

        #define S(point) textureSampleDenormalized%(textureUnit)s(volumeTextureUnit, point)

        // read from 3D texture
        sample = S(stpPoint);

        // central difference sample gradient (P is +1, N is -1)
        float sP00 = S(stpPoint + vec3(%(mmToS)f * gradientSize,0,0));
        float sN00 = S(stpPoint - vec3(%(mmToS)f * gradientSize,0,0));
        float s0P0 = S(stpPoint + vec3(0,%(mmToT)f * gradientSize,0));
        float s0N0 = S(stpPoint - vec3(0,%(mmToT)f * gradientSize,0));
        float s00P = S(stpPoint + vec3(0,0,%(mmToP)f * gradientSize));
        float s00N = S(stpPoint - vec3(0,0,%(mmToP)f * gradientSize));

        #undef S

        // TODO: add Sobel option to filter gradient
        // https://en.wikipedia.org/wiki/Sobel_operator#Extension_to_other_dimensions

        vec3 gradient = vec3( (sP00-sN00),
                              (s0P0-s0N0),
                              (s00P-s00N) );

        gradientMagnitude = length(gradient);

        // https://en.wikipedia.org/wiki/Normal_(geometry)#Transforming_normals
        mat3 normalSTPToRAS = mat3(%(normalSTPToRAS)s);
        vec3 localNormal;
        localNormal = (-1. / gradientMagnitude) * gradient;
        normal = normalize(normalSTPToRAS * localNormal);
      }
    """ % sampleVolumeParameters
    return transferFunctionSource + fieldSampleSource

class SceneRenderer(object):
  """A class to render the current mrml scene as using shader computation"""

  def __init__(self, scene=None):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    self.volumeTexture = None
    self.logic = ShaderComputationLogic()
    try:
      from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLShaderComputation
    except ImportError:
      import vtkAddon
      vtkOpenGLShaderComputation=vtkAddon.vtkOpenGLShaderComputation
    self.shaderComputation=vtkOpenGLShaderComputation()

    self.shaderComputation.SetVertexShaderSource(self.logic.rayCastVertexShaderSource())

    self.resultImage = vtk.vtkImageData()
    self.resultImage.SetDimensions(512, 512, 1)
    self.resultImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    self.shaderComputation.SetResultImageData(self.resultImage)

    self.imageViewer = vtk.vtkImageViewer()
    self.imageViewer.SetColorLevel(128)
    self.imageViewer.SetColorWindow(256)

    self.transformPointSource = "" ;# TODO: this is just an example
    self.volumePropertyNode = None

    # map of node IDs to instances of FieldSampler subclasses
    self.fieldSamplersByNodeID = {}

    if scene:
      self.scene = scene
    else:
      self.scene = slicer.mrmlScene
    self.sceneObserver = SceneObserver(self.scene)
    self.sceneObserver.addTrigger("ScalarVolume", "Added", self.onVolumeAdded)
    self.sceneObserver.addTrigger("ScalarVolume", "Removed", self.onVolumeRemoved)
    self.sceneObserver.addTrigger("ScalarVolume", "Modified", self.requestRender)
    self.sceneObserver.addTrigger("ScalarVolume", "ImageDataModified", self.requestRender)
    self.sceneObserver.addTrigger("LabelMapVolume", "Modified", self.requestRender)
    self.sceneObserver.addTrigger("LabelMapVolume", "ImageDataModified", self.requestRender)
    self.sceneObserver.addTrigger("Camera", "Modified", self.requestRender)
    self.sceneObserver.addTrigger("LinearTransform", "Modified", self.requestRender)
    self.sceneObserver.addTrigger("VolumeProperty", "Modified", self.requestRender)
    self.sceneObserver.addTrigger("MarkupsFiducial", "Modified", self.requestRender)
    self.sceneObserver.addTrigger("ScalarVolume", "No", self.requestRender) # sent from transform modified
    self._renderPending = False
    self._active = True # ignore events when not active

  def cleanup(self):
    self.sceneObserver.removeObservers()
    self.sceneObserver = None
    self.volumeTexture = None
    self.shaderComputation = None
    self.imageViewer = None

  def deactivate(self):
    self._active = False

  def activate(self):
    self._active = True
    self.updateFieldSamplers()
    self.render()

  def setVolume(self, volumeNode):
    self.volumeTexture = VolumeTexture(self.shaderComputation, 15, volumeNode)
    self.updateFieldSamplers()

  def onVolumeAdded(self):
    self.updateFieldSamplers()

  def onVolumeRemoved(self):
    self.updateFieldSamplers()

  def updateAllocatedTextureUnits(self):
    self.allocatedTextureUnits = []
    for fieldSamplers in self.fieldSamplersByNodeID.values():
      self.allocatedTextureUnits.append(fieldSamplers.textureUnit)

  def getFreeTextureUnit(self):
    for unit in range(48):
      if not unit in self.allocatedTextureUnits:
        self.allocatedTextureUnits.append(unit)
        return unit
    logging.error('no texture units available')
    return -1

  def updateFieldSamplers(self):
    """For now, hard code the mapping from nodes to FieldSampler, but eventually
    consider making plugins that handle various node types"""
    if not self._active:
      return
    self.updateAllocatedTextureUnits()
    mappedNodeIDs = []
    slicer.mrmlScene.InitTraversal()
    node = slicer.mrmlScene.GetNextNode()
    while node:
      id_ = node.GetID()
      if id_ in self.fieldSamplersByNodeID.keys():
        mappedNodeIDs.append(id_)
      else:
        if node.GetClassName() == 'vtkMRMLScalarVolumeNode':
          textureUnit = self.getFreeTextureUnit()
          self.fieldSamplersByNodeID[id_] = VolumeTexture(self.shaderComputation, textureUnit, node)
          mappedNodeIDs.append(id_)
          print(id_, node.GetName(), 'mapped as', textureUnit)
        # TODO: implement label map field
        if False and node.GetClassName() == 'vtkMRMLLabelMapVolumeNode':
          textureUnit = self.getFreeTextureUnit()
          self.fieldSamplersByNodeID[id_] = LabelMapTexture(self.shaderComputation, textureUnit, node)
          mappedNodeIDs.append(id_)
          print(id_, node.GetName(), 'mapped as', textureUnit)
        elif node.GetClassName() == 'vtkMRMLMarkupsFiducialNode':
          textureUnit = self.getFreeTextureUnit()
          self.fieldSamplersByNodeID[id_] = Fiducials(self.shaderComputation, textureUnit, node)
          mappedNodeIDs.append(id_)
          print(id_, node.GetName(), 'mapped as', textureUnit)
      node = slicer.mrmlScene.GetNextNode()

    for id_ in self.fieldSamplersByNodeID.keys():
      if not id_ in mappedNodeIDs:
        del(self.fieldSamplersByNodeID[id_])
        print(id_, 'removed')

  def fieldSamplersSource(self):
    """Functions to sample all currently mapped nodes"""
    samplersSource = ''
    for fieldSampler in self.fieldSamplersByNodeID.values():
      samplersSource += fieldSampler.fieldSampleSource()
    return samplersSource

  def fieldCompositeSource(self):
    """Inner loop to composite all currently mapped nodes (used in ray casting)"""

    fieldSampleTemplate = """
          // accumulate per-field opacities and lit colors
          sampleVolume%(textureUnit)s(textureUnit%(textureUnit)s, samplePoint, gradientSize, sample, normal, gradientMagnitude);

          transferFunction%(textureUnit)s(sample, gradientMagnitude, color, fieldOpacity);

          litColor += fieldOpacity * lightingModel(samplePoint, normal, color, eyeRayOrigin);
          opacity += fieldOpacity;
    """

    fieldCompositeSource =  """
          vec3 normal;
          float gradientMagnitude;
          vec3 color;
          float opacity = 0.;
          vec3 litColor = vec3(0.);
          float fieldOpacity = 0.;
          vec3 fieldLitColor = vec3(0.);
    """

    for fieldSampler in self.fieldSamplersByNodeID.values():
      fieldCompositeSource += fieldSampleTemplate % {
              'textureUnit' : fieldSampler.textureUnit
      }

    fieldCompositeSource +=  """
        // normalize back so that litColor is mean of all fields weighted by opacity
        litColor /= opacity;
    """

    return fieldCompositeSource


  def render(self):
    """Perform the actual render operation by pulling together all
    the elements of the shader program and ensuring the data is up to
    date on the GPU.  Compute the result and display in a window."""

    if not self._active:
      return

    if not self.shaderComputation:
      logging.error("can't render without computation context")
      return

    self.updateFieldSamplers()

    if len(self.fieldSamplersByNodeID.values()) == 0:
      logging.error ("can't render without fields")
      return

    for fieldSampler in self.fieldSamplersByNodeID.values():
      fieldSampler.updateFromMRML()

    # need to declare each texture unit as a uniform passed in
    # from the host code; these are done in the vtkOpenGLTextureImage instances
    textureUnitDeclaration = "uniform sampler3D "
    for fieldSampler in self.fieldSamplersByNodeID.values():
      textureUnitDeclaration += "textureUnit%d," % fieldSampler.textureUnit
    textureUnitDeclaration = textureUnitDeclaration[:-1] + ';'

    if not self.volumeTexture:
      volumeNode = slicer.util.getNode('vtkMRMLScalarVolumeNode*')
      if not volumeNode:
        return
      self.volumeTexture = VolumeTexture(self.shaderComputation, 15, volumeNode)

    rayCastParameters = self.logic.rayCastVolumeParameters(self.volumeTexture.node)
    rayCastParameters.update({
          'rayMaxSteps' : 100000,
          'compositeSource' : self.fieldCompositeSource()
    })
    rayCastSource = self.logic.rayCastFragmentSource() % rayCastParameters

    self.shaderComputation.SetFragmentShaderSource("""
      %(header)s
      %(textureUnitDeclaration)s
      %(intersectBox)s
      %(transformPoint)s
      %(fieldSamplers)s
      %(rayCast)s

      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        gl_FragColor = rayCast(interpolatedTextureCoordinate);
      }
    """ % {
      'header' : self.logic.headerSource(),
      'textureUnitDeclaration' : textureUnitDeclaration,
      'intersectBox' : self.logic.intersectBoxSource(),
      'transformPoint' : self.transformPointSource,
      'fieldSamplers' : self.fieldSamplersSource(),
      'rayCast' : rayCastSource,
    })

    self.shaderComputation.AcquireResultRenderbuffer()
    self.shaderComputation.Compute()
    self.shaderComputation.ReadResult()
    self.shaderComputation.ReleaseResultRenderbuffer()

    # copy result to the image viewer
    self.imageViewer.SetInputData(self.resultImage)
    self.imageViewer.Render()

    self._renderPending = False

    if False:
      # print(self.shaderComputation.GetFragmentShaderSource())
      fp = open('/tmp/shader.glsl','w')
      fp.write(self.shaderComputation.GetFragmentShaderSource())
      fp.close()

  def requestRender(self):
    if not self._renderPending:
      self._renderPending = True
      qt.QTimer.singleShot(0,self.render)


#
# ShaderComputationLogic
#

class ShaderComputationLogic(ScriptedLoadableModuleLogic):
  """Helper methods to map from slicer/vtk conventions
  to glsl code.
  """

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    # TODO: these strings can move to a CommonGL spot once debugged

  def volumeRenderingDisplayNode(self,volumeNode):
    """Get the display node or create a new one if missing"""
    displayNode = None
    displayNodeCount = volumeNode.GetNumberOfDisplayNodes()
    for displayNodeIndex in range(displayNodeCount):
      thisDisplayNode = volumeNode.GetNthDisplayNode(displayNodeIndex)
      if thisDisplayNode.GetClassName() == 'vtkMRMLGPURayCastVolumeRenderingDisplayNode':
          displayNode = thisDisplayNode
    if not displayNode:
      displayNode = ShaderComputationLogic().createVolumeDisplayNode(volumeNode, (1,1,1))
    return displayNode

  def createVolumeDisplayNode(self,volumeNode,baseColor=(1,1,1)):
    """Adds a volume rendering display node to a volume node"""

    # create the volume property node with default transfer functions
    volumePropertyNode = slicer.vtkMRMLVolumePropertyNode()
    volumePropertyNode.SetName(volumeNode.GetName() + '-VP')
    scalarOpacity = vtk.vtkPiecewiseFunction()
    volumePropertyNode.SetScalarOpacity(scalarOpacity)
    gradientOpacity = vtk.vtkPiecewiseFunction()
    volumePropertyNode.SetGradientOpacity(gradientOpacity)
    colorTransfer = vtk.vtkColorTransferFunction()
    volumePropertyNode.SetColor(colorTransfer, 0)
    slicer.mrmlScene.AddNode(volumePropertyNode)
    # create the display node and give it the volume property
    displayNode = slicer.vtkMRMLGPURayCastVolumeRenderingDisplayNode()
    displayNode.SetName(volumeNode.GetName() + '-VR')
    displayNode.SetAndObserveVolumePropertyNodeID(volumePropertyNode.GetID())
    displayNode.SetVisibility(1)
    displayNode.SetAndObserveVolumeNodeID(volumeNode.GetID())
    slicer.mrmlScene.AddNode(displayNode)
    volumeNode.AddAndObserveDisplayNodeID(displayNode.GetID())

    # guess the transfer function based on the volume data
    scalarRange = volumeNode.GetImageData().GetScalarRange()
    rangeWidth = scalarRange[1] - scalarRange[0]
    rangeCenter = scalarRange[0] + rangeWidth * 0.5
    scalarOpacityPoints = (
            (scalarRange[0], 0.),
            (rangeCenter - 0.1 * rangeWidth, 0.),
            (rangeCenter + 0.1 * rangeWidth, 1.),
            (scalarRange[1], 1.) )
    for point in scalarOpacityPoints:
      scalarOpacity.AddPoint(*point)
    gradientOpacityPoints = (
            (0, 0.),
            (rangeCenter - 0.1 * rangeWidth, 0.),
            (rangeCenter + 0.1 * rangeWidth, 1.),
            (scalarRange[1], 1.) )
    for point in gradientOpacityPoints:
      gradientOpacity.AddPoint(*point)
    colorPoints = (
            (scalarRange[0], baseColor),
            (scalarRange[1], baseColor) )
    for intensity,rgb in colorPoints:
      colorTransfer.AddRGBPoint(intensity, *rgb)

    return displayNode

  def headerSource(self):
    return ("""
      #version 120
    """)

  def rayCastVolumeParameters(self,volumeNode):
    """Calculate the dictionary of substitutions for the
    current state of the volume node and camera in a form for
    substitution into the rayCast shader function
    TODO: this would probably be better as uniforms, but that
    requires doing a lot of parsing and data management in C++
    """

    parameters = {}

    rasBounds = [0,]*6
    volumeNode.GetRASBounds(rasBounds)
    rasBoxMin = (rasBounds[0], rasBounds[2], rasBounds[4])
    rasBoxMax = (rasBounds[1], rasBounds[3], rasBounds[5])
    parameters['rasBoxMin'] = "%f, %f, %f" % rasBoxMin
    parameters['rasBoxMax'] = "%f, %f, %f" % rasBoxMax

    # conservative guesses:
    # sampleStep is in mm, shortest side in world space divided by max volume dimension
    # gradientSize is in [0,1] texture space, sampleStep divided by max volume dimensions

    rasMinSide = min(rasBoxMax[0] - rasBoxMin[0], rasBoxMax[1] - rasBoxMin[1], rasBoxMax[2] - rasBoxMin[2])
    maxDimension = max(volumeNode.GetImageData().GetDimensions())
    parameters['sampleStep'] = 1. * rasMinSide / maxDimension
    minSpacing = min(volumeNode.GetSpacing())
    parameters['gradientSize'] = .5 * minSpacing

    # get the camera parameters from default 3D window
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    renderWindow = threeDView.renderWindow()
    renderers = renderWindow.GetRenderers()
    renderer = renderers.GetItemAsObject(0)
    camera = renderer.GetActiveCamera()

    import math
    import numpy
    viewPosition = numpy.array(camera.GetPosition())
    focalPoint = numpy.array(camera.GetFocalPoint())
    viewDistance = numpy.linalg.norm(focalPoint - viewPosition)
    viewNormal = (focalPoint - viewPosition) / viewDistance
    viewAngle = camera.GetViewAngle()
    viewUp = numpy.array(camera.GetViewUp())
    viewRight = numpy.cross(viewNormal,viewUp)

    parameters['eyeRayOrigin'] = "%f, %f, %f" % (viewPosition[0], viewPosition[1], viewPosition[2])
    parameters['viewNormal'] = "%f, %f, %f" % (viewNormal[0], viewNormal[1], viewNormal[2])
    parameters['viewRight'] = "%f, %f, %f" % (viewRight[0], viewRight[1], viewRight[2])
    parameters['viewUp'] = "%f, %f, %f" % (viewUp[0], viewUp[1], viewUp[2])
    parameters['halfSinViewAngle'] = "%f" % (0.5 * math.cos(math.radians(viewAngle)))

    return parameters

  def rayCastVertexShaderSource(self):
    return ("""
      %(header)s
      attribute vec3 vertexAttribute;
      attribute vec2 textureCoordinateAttribute;
      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        interpolatedTextureCoordinate = vec3(textureCoordinateAttribute, .5);
        gl_Position = vec4(vertexAttribute, 1.);
      }
    """ % {
      'header' : self.headerSource()
    })

  def rayCastFragmentSource(self):
    return("""
      vec3 lightingModel( in vec3 samplePoint, in vec3 normal, in vec3 color, in vec3 eyeRayOrigin )
      {
        // Phong lighting
        // http://en.wikipedia.org/wiki/Phong_reflection_model
        vec3 Cambient = color;
        vec3 Cdiffuse = color;
        vec3 Cspecular = vec3(1.,1.,1.);
        float Kambient = .30;
        float Kdiffuse = .95;
        float Kspecular = .90;
        float Shininess = 15.;
        vec3 pointLight = vec3(200., 2500., 1000.); // TODO - lighting model

        vec3 litColor = Kambient * Cambient;
        vec3 pointToEye = normalize(eyeRayOrigin - samplePoint);

        if (dot(pointToEye, normal) > 0.) {
          vec3 pointToLight = normalize(pointLight - samplePoint);
          float lightDot = dot(pointToLight,normal);
          vec3 lightReflection = reflect(pointToLight,normal);
          float reflectDot = dot(lightReflection,pointToEye);
          if (lightDot > 0.) {
            litColor += Kdiffuse * lightDot * Cdiffuse;
            litColor += Kspecular * pow( reflectDot, Shininess ) * Cspecular;
          }
        }
        return litColor;
      }

      // field ray caster - starts from the front and collects color and opacity
      // contributions until fully saturated.
      // Sample coordinate is 0->1 texture space
      vec4 rayCast( in vec3 sampleCoordinate )
      {
        vec4 backgroundRGBA = vec4(0.,0.,.5,1.); // TODO: mid blue background for now

        // TODO aspect: float aspect = imageW / (1.0 * imageH);
        vec2 normalizedCoordinate = 2. * (sampleCoordinate.st -.5);

        // calculate eye ray in world space
        vec3 eyeRayOrigin = vec3(%(eyeRayOrigin)s);
        vec3 eyeRayDirection;

        // ||viewNormal + u * viewRight + v * viewUp||

        eyeRayDirection = normalize (                            vec3( %(viewNormal)s )
                                      + ( %(halfSinViewAngle)s * normalizedCoordinate.x * vec3( %(viewRight)s ) )
                                      + ( %(halfSinViewAngle)s * normalizedCoordinate.y * vec3( %(viewUp)s    ) ) );



        // find intersection with box, possibly terminate early
        float tNear, tFar;
        vec3 rasBoxMin = vec3( %(rasBoxMin)s );
        vec3 rasBoxMax = vec3( %(rasBoxMax)s );
        bool hit = intersectBox( eyeRayOrigin, eyeRayDirection, rasBoxMin, rasBoxMax, tNear, tFar );
        if (!hit) {
          return (backgroundRGBA);
        }

        if (tNear < 0.) tNear = 0.;     // clamp to near plane

        // march along ray from front, accumulating color and opacity
        vec4 integratedPixel = vec4(0.);
        float gradientSize = %(gradientSize)f;
        float tCurrent = tNear;
        float sample;
        int rayStep;
        for(rayStep = 0; rayStep < %(rayMaxSteps)d; rayStep++) {

          vec3 samplePoint = eyeRayOrigin + eyeRayDirection * tCurrent;

          %(compositeSource)s

          // http://graphicsrunner.blogspot.com/2009/01/volume-rendering-101.html
          opacity *= %(sampleStep)f;
          integratedPixel.rgb += (1. - integratedPixel.a) * opacity * litColor;
          integratedPixel.a += (1. - integratedPixel.a) * opacity;
          integratedPixel = clamp(integratedPixel, 0., 1.);

          tCurrent += %(sampleStep)f;
          if (
              tCurrent >= tFar  // stepped out of the volume
                ||
              integratedPixel.a >= 1.  // pixel is saturated
          ) {
            break; // we can stop now
          }
        }
        return(vec4(mix(backgroundRGBA.rgb, integratedPixel.rgb, integratedPixel.a), 1.));
      }
    """)

  def intersectBoxSource(self):
    return ("""
      bool intersectBox(const in vec3 rayOrigin, const in vec3 rayDirection,
                        const in vec3 boxMin, const in vec3 boxMax,
                        out float tNear, out float tFar)
        // intersect ray with a box
        // http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
      {
          // compute intersection of ray with all six bbox planes
          vec3 invRay = vec3(1.) / rayDirection;
          vec3 tBot = invRay * (boxMin - rayOrigin);
          vec3 tTop = invRay * (boxMax - rayOrigin);

          // re-order intersections to find smallest and largest on each axis
          vec3 tMin = min(tTop, tBot);
          vec3 tMax = max(tTop, tBot);

          // find the largest tMin and the smallest tMax
          float largest_tMin = max(max(tMin.x, tMin.y), max(tMin.x, tMin.z));
          float smallest_tMax = min(min(tMax.x, tMax.y), min(tMax.x, tMax.z));

          tNear = largest_tMin;
          tFar = smallest_tMax;

          return smallest_tMax > largest_tMin;
      }
    """)


class ShaderComputationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    pass


  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setup()
    self.test_ShaderComputation()

  def addDefaultVolumeProperty(self):
    # create a volume property node in the scene if needed.  This is used
    # for the color transfer function and can be manipulated in the
    # Slicer Volume Rendering module widget
    volumePropertyNode = slicer.util.getNode('ShaderVolumeProperty')
    if not volumePropertyNode:
      volumePropertyNode = slicer.vtkMRMLVolumePropertyNode()
      volumePropertyNode.SetName('ShaderVolumeProperty')
      scalarOpacity = vtk.vtkPiecewiseFunction()
      points = ( (-1024., 0.), (20., 0.), (300., 1.), (3532., 1.) )
      for point in points:
        scalarOpacity.AddPoint(*point)
      volumePropertyNode.SetScalarOpacity(scalarOpacity)
      colorTransfer = vtk.vtkColorTransferFunction()
      colors = ( (-1024., (0., 0., 0.)), (3., (0., 0., 0.)), (131., (1., 1., 0.)) )
      colors = ( (-1024., (0., 0., 0.)), (-984., (0., 0., 0.)), (469., (1., 1., 1.)) )
      for intensity,rgb in colors:
        colorTransfer.AddRGBPoint(intensity, *rgb)
      volumePropertyNode.SetScalarOpacity(scalarOpacity)
      volumePropertyNode.SetColor(colorTransfer, 0)
      slicer.mrmlScene.AddNode(volumePropertyNode)
    self.volumePropertyNode = volumePropertyNode

  def addVolume(self):
    """unless given a volume in the constructor make sure
    there is a volume and set the instance variable"""
    import SampleData
    sampleDataLogic = SampleData.SampleDataLogic()
    name, method = 'MRHead', sampleDataLogic.downloadMRHead
    name, method = 'CTACardio', sampleDataLogic.downloadCTACardio
    volumeToRender = slicer.util.getNode(name)
    if not volumeToRender:
      logging.info("Getting Volume %s" % name)
      volumeToRender = method()
    ShaderComputationLogic().createVolumeDisplayNode(volumeToRender, (1,1,1))

  def amigoMRUSData(self):
    preopT2orig = 'https://docs.google.com/uc?authuser=1&id=0Bygzw56l1ZC-Q0dCLThIelVVaFE&export=download'
    preopT2smooth = 'https://docs.google.com/uc?authuser=1&id=0Bygzw56l1ZC-Q0dCLThIelVVaFE&export=download'
    source = ( 'MR-US Neuro',
              ('https://docs.google.com/uc?authuser=1&id=0Bygzw56l1ZC-MFp5RGZMSmF4YVk&export=download',
               'https://docs.google.com/uc?authuser=1&id=0Bygzw56l1ZC-UmhGUk51NXdpZ3M&export=download',
               preopT2smooth,
               'https://docs.google.com/uc?authuser=1&id=0Bygzw56l1ZC-MGVUX2QyRllPcW8&export=download'
              ),
              ('intra-T2.nrrd', 'intra-US.nrrd', 'preop-T2.nrrd', 'preop-US.nrrd'),
              ('intra-T2', 'intra-US','preop-T2', 'preop-US')
            )
    import SampleData
    SampleData.SampleDataLogic().downloadFromSource(SampleData.SampleDataSource(*source))

  def amigoMRUSPreIntraData(self):
    preopT2orig = 'https://docs.google.com/uc?authuser=1&id=0Bygzw56l1ZC-Q0dCLThIelVVaFE&export=download'
    preopT2smooth = 'https://docs.google.com/uc?authuser=1&id=0Bygzw56l1ZC-Q0dCLThIelVVaFE&export=download'
    preopT2smoothN4 = 'https://drive.google.com/drive/u/0/folders/0Bygzw56l1ZC-QzBkYktZa3RhazA'
    source = ( 'MR-US Neuro PreIntra',
              ('https://docs.google.com/uc?authuser=1&id=0Bygzw56l1ZC-UmhGUk51NXdpZ3M&export=download',
               preopT2smoothN4
              ),
              ('intra-US.nrrd', 'preop-T2-smooth.nrrd'),
              ('intra-US','preop-T2')
            )
    import SampleData
    return SampleData.SampleDataLogic().downloadFromSource(SampleData.SampleDataSource(*source))

  def amigoScenario(self):
    """Load MR and US data to emulate intraprocedural imaging"""
    # self.amigoMRUSData()
    nodes = self.amigoMRUSPreIntraData()
    ShaderComputationLogic().createVolumeDisplayNode(nodes[0], (0,1,0))
    ShaderComputationLogic().createVolumeDisplayNode(nodes[1], (1,1,1))

  def addFiducials(self):
    shaderFiducials = slicer.util.getNode('shaderFiducials')
    if not shaderFiducials:
      displayNode = slicer.vtkMRMLMarkupsDisplayNode()
      slicer.mrmlScene.AddNode(displayNode)
      fiducialNode = slicer.vtkMRMLMarkupsFiducialNode()
      fiducialNode.SetName('shaderFiducials')
      slicer.mrmlScene.AddNode(fiducialNode)
      fiducialNode.SetAndObserveDisplayNodeID(displayNode.GetID())
      for ras in ((28.338526, 34.064367, 10), (-10, 0, -5)):
        fiducialNode.AddFiducial(*ras)
      import random
      fiducialCount = 10
      radius = 75
      for index in range(fiducialCount):
        uvw = [random.random(), random.random(), random.random()]
        ras = map(lambda e: radius * (2. * e - 1.), uvw)
        fiducialNode.AddFiducial(*ras)

      # make it active
      selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
      if (selectionNode is not None):
        selectionNode.SetReferenceActivePlaceNodeID(fiducialNode.GetID())

  def test_ShaderComputation(self):
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

    slicer.modules.ShaderComputationWidget.sceneRenderer.deactivate()
    self.addDefaultVolumeProperty()
    scenario = 'chest'
    scenario = 'amigo'
    if scenario == 'amigo':
      self.amigoScenario()
    else:
      self.addFiducials()
      self.addVolume()
    slicer.modules.ShaderComputationWidget.sceneRenderer.activate()
