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
    This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
    and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
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
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

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
    logic = ShaderComputationLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

#
# ShaderComputationLogic
#

class ShaderComputationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)


class ShaderComputationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    # slicer.mrmlScene.Clear(0)
    pass

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    #self.test_ShaderComputation1()
    self.test_ShaderComputation2()

  def test_ShaderComputation1(self):
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

    # self.delayDisplay("Starting the test", 100)

    mrHeadVolume = slicer.util.getNode('MRHead')
    if not mrHeadVolume:
      import SampleData
      sampleDataLogic = SampleData.SampleDataLogic()
      print("Getting MR Head Volume")
      mrHeadVolume = sampleDataLogic.downloadMRHead()

    resize = vtk.vtkImageResize()
    resize.SetInputDataObject(mrHeadVolume.GetImageData())
    resize.SetOutputDimensions(256,256,128)
    resize.Update()

    from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLShaderComputation

    shaderComputation=vtkOpenGLShaderComputation()

    shaderComputation.SetVertexShaderSource("""
      #version 120
      attribute vec3 vertexAttribute;
      attribute vec2 textureCoordinateAttribute;
      varying vec4 interpolatedColor;
      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        interpolatedColor = vec4(0.5) + vec4(vertexAttribute, 1.);
        interpolatedTextureCoordinate = vec3(textureCoordinateAttribute, .5);
        gl_Position = vec4(vertexAttribute, 1.);
      }
    """)

    shaderComputation.SetFragmentShaderSource("""
      #version 120
      varying vec4 interpolatedColor;
      varying vec3 interpolatedTextureCoordinate;
      uniform sampler3D volumeSampler;
      void main()
      {
      /*
        gl_FragColor = vec4 ( 1.0, 0.0, 0.0, 1.0 );
        gl_FragColor = interpolatedColor;
        vec4 volumeSample = texture3D(volumeSampler, interpolatedTextureCoordinate);
        volumeSample *= 100;
        gl_FragColor = mix( volumeSample, interpolatedColor, 0.5);
      */

        vec4 integratedRay = vec4(0.);
        for (int i = 0; i < 256; i++) {
          vec3 samplePoint = vec3(interpolatedTextureCoordinate.st, i/256.);
          vec4 volumeSample = texture3D(volumeSampler, samplePoint);
          integratedRay += volumeSample;
        }
        gl_FragColor = integratedRay;
      }
    """)
    shaderComputation.SetTextureImageData(resize.GetOutputDataObject(0))

    resultImage = vtk.vtkImageData()
    resultImage.SetDimensions(512, 512, 1)
    resultImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    shaderComputation.SetResultImageData(resultImage)

    shaderComputation.Compute()

    iv = vtk.vtkImageViewer()
    iv.SetColorLevel(128)
    iv.SetColorWindow(256)
    iv.SetInputData(resultImage)
    iv.Render()

    slicer.modules.ShaderComputationWidget.iv = iv

  def sampleVolumeParameters(self,volumeNode):
    """Calculate the dictionary of substitutions for the
    current state of the volume node in a form for
    substitution into the sampleVolume shader function
    TODO: this would probably be better as uniforms, but that
    requires doing a lot of parsing and data management in C++
    """

    volumeArray = slicer.util.array(volumeNode.GetID())

    rasToIJK = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIJK)

    transformNode = volumeNode.GetParentTransformNode()
    if transformNode:
      if transformNode.IsTransformToWorldLinear():
        rasToRAS = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToWorld(rasToRAS)
        rasToRAS.Invert()
        rasToRAS.Multiply4x4(rasToIJK, rasToRAS, rasToIJK)
      else:
        print('Cannot handle nonlinear transforms')

    # dimensions are number of pixels in (row, column, slice)
    # which maps to 0-1 space of S, T, P
    dimensions = volumeNode.GetImageData().GetDimensions()
    ijkToSTP = vtk.vtkMatrix4x4()
    ijkToSTP.Identity()
    for diagonal in range(3):
      ijkToSTP.SetElement(diagonal,diagonal, 1./dimensions[diagonal])
    ijkToSTP.Multiply4x4(ijkToSTP, rasToIJK, rasToIJK)

    parameters = {}
    rows = ('rasToS', 'rasToT', 'rasToP')
    for row in range(3):
      rowKey = rows[row]
      parameters[rowKey] = ""
      for col in range(4):
        element = rasToIJK.GetElement(row,col)
        parameters[rowKey] += "%f," % element
      parameters[rowKey] = parameters[rowKey][:-1] # clear trailing comma

    parameters['scalarTypeMax'] = 32767. # TODO: get this from scalar type

    return parameters

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
    # gradientSize is in [0,1] texture sapce, sampleStep divided by max volume dimensions

    rasMinSide = min(rasBoxMax[0] - rasBoxMin[0], rasBoxMax[1] - rasBoxMin[1], rasBoxMax[2] - rasBoxMin[2])
    maxDimension = max(volumeNode.GetImageData().GetDimensions())
    parameters['sampleStep'] = .25 * rasMinSide / maxDimension
    parameters['gradientSize'] = parameters['sampleStep'] / maxDimension

    # get the camera parameters from default 3D window
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    renderWindow = threeDView.renderWindow()
    renderers = renderWindow.GetRenderers()
    renderer = renderers.GetItemAsObject(0)
    camera = renderer.GetActiveCamera()

    import numpy
    viewPosition = numpy.array(camera.GetPosition())
    focalPoint = numpy.array(camera.GetFocalPoint())
    viewDistance = numpy.linalg.norm(focalPoint - viewPosition)
    viewNormal = (focalPoint - viewPosition) / viewDistance
    viewUp = numpy.array(camera.GetViewUp())
    viewAngle = camera.GetViewAngle()
    viewRight = numpy.cross(viewNormal,viewUp)

    parameters['eyeRayOrigin'] = "%f, %f, %f" % (viewPosition[0], viewPosition[1], viewPosition[2])
    parameters['viewNormal'] = "%f, %f, %f" % (viewNormal[0], viewNormal[1], viewNormal[2])
    parameters['viewRight'] = "%f, %f, %f" % (viewRight[0], viewRight[1], viewRight[2])
    parameters['viewUp'] = "%f, %f, %f" % (viewUp[0], viewUp[1], viewUp[2])

    return parameters

  def transferFunctionSource(self, volumePropertyNode):
    """Create source code for transfer function that maps
    a sample and gradient to a color and opacity based on
    the passed volumePropertyNode.
    """
    scalarOpacity = volumePropertyNode.GetScalarOpacity(0)
    colorTransfer = volumePropertyNode.GetColor(0)

    source = """
    void transferFunction(const in float sample, const in float gradientMagnitude,
                          out vec3 color, out float opacity)
    {
    """

    # convert the scalarOpacity transfer function to a procedure
    # - ignore the interpolation options; only linear interpolation
    intensities = []
    opacities = []
    size = scalarOpacity.GetSize()
    values = [0,]*4
    for index in range(size):
      scalarOpacity.GetNodeValue(index, values)
      intensities.append(values[0])
      opacities.append(values[1])
    source += """
      if (sample < %(minIntensity)f) {
        opacity = %(minOpacity)f;
      }
    """ % {'minIntensity': intensities[0], 'minOpacity': opacities[0]}
    for index in range(size-1):
      currentIndex = index + 1
      source += """
        else if (sample < %(currentIntesity)f) {
          opacity = mix(%(lastOpacity)f, %(currentOpacity)f, (sample - %(lastIntensity)f) / %(intensityRange)f);
        }
      """ % {'currentIntesity': intensities[currentIndex],
             'lastOpacity': opacities[index],
             'currentOpacity': opacities[currentIndex],
             'lastIntensity': intensities[index],
             'intensityRange': intensities[currentIndex] - intensities[index],
             }
    source += """
      else {
        opacity = %(lastOpacity)f;
      }
    """ % {'lastOpacity': opacities[size-1]}

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


  def test_ShaderComputation2(self, caller=None, event=None):
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

    import SampleData
    sampleDataLogic = SampleData.SampleDataLogic()
    name, method = 'MRHead', sampleDataLogic.downloadMRHead
    name, method = 'CTACardio', sampleDataLogic.downloadCTACardio
    volumeToRender = slicer.util.getNode(name)
    if not volumeToRender:
      print("Getting Volume")
      volumeToRender = method()

    if False:
      ellipsoid = vtk.vtkImageEllipsoidSource()
      ellipsoid.SetInValue(255)
      ellipsoid.SetOutValue(0)
      ellipsoid.SetOutputScalarTypeToShort()
      ellipsoid.SetWholeExtent(0, 255, 0, 255, 0, 255)
      ellipsoid.Update()
      volumeToRender.SetAndObserveImageData(ellipsoid.GetOutputDataObject(0))

    if not hasattr(self,"shaderComputation") and hasattr(slicer.modules.ShaderComputationInstance, "test"):
      oldSelf = slicer.modules.ShaderComputationInstance.test
      oldSelf.renderWindow.RemoveObserver(oldSelf.renderTag)

    if not hasattr(self,"shaderComputation"):
      print('new shaderComputation')
      from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLShaderComputation
      self.shaderComputation=vtkOpenGLShaderComputation()

    # TODO: these strings can move to a CommonGL spot once debugged
    headerSource = """
      #version 120
    """

    self.shaderComputation.SetVertexShaderSource("""
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
      'header' : headerSource
    })


    intersectBoxSource = """
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
    """

    sampleVolumeSource = """
      float textureSampleDenormalized(const in sampler3D volumeSampler, const in vec3 stpPoint) {
        return ( texture3D(volumeSampler, stpPoint).r * %(scalarTypeMax)f );
      }

      void sampleVolume(const in sampler3D volumeSampler, const in vec3 samplePoint, const in float gradientSize,
                        out float sample, out vec3 normal, out float gradientMagnitude)
      {
        // vectors to map RAS to stp
        vec4 rasToS =  vec4( %(rasToS)s );
        vec4 rasToT =  vec4( %(rasToT)s );
        vec4 rasToP =  vec4( %(rasToP)s );

        vec3 stpPoint;
        vec4 sampleCoordinate = vec4(samplePoint, 1.);
        stpPoint.s = dot(rasToS,sampleCoordinate);
        stpPoint.t = dot(rasToT,sampleCoordinate);
        stpPoint.p = dot(rasToP,sampleCoordinate);

        // read from 3D texture
        sample = textureSampleDenormalized(volumeSampler, stpPoint);

        // central difference sample gradient (N is -1)
        float s100 = textureSampleDenormalized(volumeSampler, stpPoint + vec3(gradientSize,0,0));
        float sN00 = textureSampleDenormalized(volumeSampler, stpPoint - vec3(gradientSize,0,0));
        float s010 = textureSampleDenormalized(volumeSampler, stpPoint + vec3(0,gradientSize,0));
        float s0N0 = textureSampleDenormalized(volumeSampler, stpPoint - vec3(0,gradientSize,0));
        float s001 = textureSampleDenormalized(volumeSampler, stpPoint + vec3(0,0,gradientSize));
        float s00N = textureSampleDenormalized(volumeSampler, stpPoint - vec3(0,0,gradientSize));


        // TODO normal should be transformed back to RAS
        vec3 gradient = vec3( (s100-sN00),
                              (s010-s0N0),
                              (s001-s00N)) / vec3(2. * gradientSize);
        gradientMagnitude = length(gradient);
        normal = (-1. / gradientMagnitude) * gradient;
      }
    """ % self.sampleVolumeParameters(volumeToRender)

    volumePropertyNode = slicer.util.getNode('ShaderVolumeProperty')
    if not volumePropertyNode:
      volumePropertyNode = slicer.vtkMRMLVolumePropertyNode()
      volumePropertyNode.SetName('ShaderVolumeProperty')
      scalarOpacity = vtk.vtkPiecewiseFunction()
      points = ( (-1024., 0.), (20., 0.), (300., .1), (3532., .1) )
      for point in points:
        scalarOpacity.AddPoint(*point)
      volumePropertyNode.SetScalarOpacity(scalarOpacity)
      colorTransfer = vtk.vtkColorTransferFunction()
      colors = ( (-1024., (0., 0., 0.)), (50., (0., 0., 0.)), (719., (1., 1., 1.)) )
      for intensity,rgb in colors:
        colorTransfer.AddRGBPoint(intensity, *rgb)
      volumePropertyNode.SetScalarOpacity(scalarOpacity)
      volumePropertyNode.SetColor(colorTransfer, 0)
      slicer.mrmlScene.AddNode(volumePropertyNode)
    transferFunctionSource = self.transferFunctionSource(volumePropertyNode)

    rayCastParameters = self.rayCastVolumeParameters(volumeToRender)
    rayCastParameters.update({
          'rayMaxSteps' : 500000,
    }) # TODO: auto calculate ray parameters
    rayCastSource = """
      // volume ray caster - starts from the front and collects color and opacity
      // contributions until fully saturated.
      // Sample coordinate is 0->1 texture space
      vec4 rayCast( in vec3 sampleCoordinate, in sampler3D volumeSampler )
      {
        // TODO aspect: float aspect = imageW / (1.0 * imageH);
        vec2 normalizedCoordinate = 2. * (sampleCoordinate.st -.5);

        // calculate eye ray in world space
        vec3 eyeRayOrigin = vec3(%(eyeRayOrigin)s);
        vec3 eyeRayDirection;

        // ||viewNormal + u * viewRight + v * viewUp||

        eyeRayDirection = normalize (                            vec3( %(viewNormal)s )
                                      + normalizedCoordinate.x * vec3( %(viewRight)s  )
                                      + normalizedCoordinate.y * vec3( %(viewUp)s     ) );


        vec3 pointLight = vec3(-250., 250., 400.); // TODO

        // find intersection with box, possibly terminate early
        float tNear, tFar;
        vec3 rasBoxMin = vec3( %(rasBoxMin)s );
        vec3 rasBoxMax = vec3( %(rasBoxMax)s );
        bool hit = intersectBox( eyeRayOrigin, eyeRayDirection, rasBoxMin, rasBoxMax, tNear, tFar );
        if (!hit) {
          return vec4(0.,0.,.5,1.); // TODO: mid blue background for now
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

          // TODO: add a vector field proportional to rayStep, u, v
          /*
          vec4 vectorField;
          float blend = rayStep * 1000. / %(rayMaxSteps)d;
          vectorField.x = blend * sin(90. * u) + cos(90. * v);
          vectorField.y = blend * -cos(90. * u) + sin(90. * v);
          vectorField.z = 0.;
          //samplePoint += vectorField;
          */

          vec3 normal;
          float gradientMagnitude;
          sampleVolume(volumeSampler, samplePoint, gradientSize, sample, normal, gradientMagnitude);


          // Phong lighting
          // http://en.wikipedia.org/wiki/Phong_reflection_model
          vec3 Cdiffuse = vec3(1.,1.,0.);
          vec3 Cspecular = vec3(1.,1.,1.);
          float Kdiffuse = .85;
          float Kspecular = .15;
          float Shininess = 55.;

          vec3 phongColor = vec3(0.);
          vec3 V = normalize(eyeRayOrigin - samplePoint);
          if (dot(V, normal) > 0.) {
            vec3 L = normalize(pointLight - samplePoint);
            vec3 R = reflect(L,normal);
            phongColor += Kdiffuse * dot(L,normal) * Cdiffuse;
            phongColor += Kspecular * pow( dot(R,V), Shininess ) * Cspecular;
          }

          /*
          vec4 color;
          color = vec4(phongColor, 1.);
          */

          // accumulate result

          float sampleEmission = %(sampleStep)f * sample / 10000.;
          float sampleOpacity = %(sampleStep)f * sample / 100.;


          vec3 color;
          float opacity = 0.;
          transferFunction(sample, gradientMagnitude, color, opacity);

          // Ray sum
          /*
          if (opacity > 0.) {
            //integratedPixel.rgb += color;
            integratedPixel.rgb += color * opacity;
          }
          */

          // gradient sum
          //integratedPixel.rgb += vec3(sqrt(gradientMagnitude) * normalize(abs(normal)) );
          //integratedPixel.rgb += vec3(sqrt(gradientMagnitude));

          integratedPixel.a += opacity;
          integratedPixel.rgb += integratedPixel.a * opacity * phongColor.rgb;



/*
          // scalar
          //integratedPixel.rgb += (1. - integratedPixel.a) * mix(vec3(sampleEmission), phongColor, 0.000005);
          integratedPixel.rgb += 0.1 * integratedPixel.a * sampleOpacity * color + 0.1 * phongColor;
          integratedPixel.a += 0.1 * sampleOpacity;

          // Phong tests
          //integratedPixel.rgb += (1. - integratedPixel.a) * mix(vec3(sampleEmission/1000.), color.rgb, 0.5);
          //integratedPixel.rgb += (1. - integratedPixel.a) * color.rgb / 1.;
          //integratedPixel.a += %(sampleStep)f * gradientMagnitude/1000.;
          //integratedPixel.a += %(sampleStep)f * sampleOpacity;
*/
          tCurrent += %(sampleStep)f;
          if (
              tCurrent >= tFar  // stepped out of the volume
                ||
              integratedPixel.a > 1.  // pixel is saturated
          ) {
            break; // we can stop now
          }
        }
        if (integratedPixel.a > 0.) {
          return(vec4(1.);
        }
        integratedPixel /= 0.1 * rayStep;
        integratedPixel = clamp(integratedPixel, 0., 1.);
        return(vec4(integratedPixel.rgb, 1.));
        //return (vec4(1. - integratedPixel.rgb, 1.));
        //return (vec4 (mix(integratedPixel.rgb, vec3(.1, .1, 0.), 1.-integratedPixel.a), 1.));
      }
    """ % rayCastParameters

    # simplified for debugging
    rayCastSource = """
      // volume ray caster - starts from the front and collects color and opacity
      // contributions until fully saturated.
      // Sample coordinate is 0->1 texture space
      vec4 rayCast( in vec3 sampleCoordinate, in sampler3D volumeSampler )
      {
        vec4 backgroundRGBA = vec4(0.,0.,.5,1.); // TODO: mid blue background for now

        // TODO aspect: float aspect = imageW / (1.0 * imageH);
        vec2 normalizedCoordinate = 2. * (sampleCoordinate.st -.5);

        // calculate eye ray in world space
        vec3 eyeRayOrigin = vec3(%(eyeRayOrigin)s);
        vec3 eyeRayDirection;

        // ||viewNormal + u * viewRight + v * viewUp||

        eyeRayDirection = normalize (                            vec3( %(viewNormal)s )
                                      + normalizedCoordinate.x * vec3( %(viewRight)s  )
                                      + normalizedCoordinate.y * vec3( %(viewUp)s     ) );


        vec3 pointLight = vec3(-250., 250., 400.); // TODO

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

          vec3 normal;
          float gradientMagnitude;
          sampleVolume(volumeSampler, samplePoint, gradientSize, sample, normal, gradientMagnitude);

          // Phong lighting
          // http://en.wikipedia.org/wiki/Phong_reflection_model
          vec3 Cdiffuse = vec3(1.,1.,0.);
          vec3 Cspecular = vec3(1.,1.,1.);
          float Kdiffuse = .85;
          float Kspecular = .15;
          float Shininess = 55.;

          vec3 phongColor = vec3(0.);
          vec3 pointToEye = normalize(eyeRayOrigin - samplePoint);
          if (dot(pointToEye, normal) > 0.) {
            vec3 pointToLight = normalize(pointLight - samplePoint);
            vec3 lightReflection = reflect(pointToLight,normal);
            phongColor += Kdiffuse * dot(pointToLight,normal) * Cdiffuse;
            phongColor += Kspecular * pow( dot(lightReflection,pointToEye), Shininess ) * Cspecular;
          }

          vec3 color;
          float opacity;
          transferFunction(sample, gradientMagnitude, color, opacity);

          if (tCurrent < 30.) {
            opacity = 0.;
          }

          integratedPixel.rgb = mix(color, integratedPixel.rgb, integratedPixel.a);
          integratedPixel.a += opacity * %(sampleStep)f;
          integratedPixel = clamp(integratedPixel, 0., 1.);

          tCurrent += %(sampleStep)f;
          if (
              tCurrent >= tFar  // stepped out of the volume
                ||
              integratedPixel.a > 1.  // pixel is saturated
          ) {
            break; // we can stop now
          }
        }
        return(vec4(mix(backgroundRGBA.rgb, integratedPixel.rgb, integratedPixel.a), 1.));
      }
    """ % rayCastParameters

    self.shaderComputation.SetFragmentShaderSource("""
      %(header)s
      %(intersectBox)s
      %(sampleVolume)s
      %(transferFunction)s
      %(rayCast)s

      varying vec3 interpolatedTextureCoordinate;
      uniform sampler3D volumeSampler;
      void main()
      {
        gl_FragColor = rayCast(interpolatedTextureCoordinate, volumeSampler);
      }
    """ % {
      'header' : headerSource,
      'intersectBox' : intersectBoxSource,
      'sampleVolume' : sampleVolumeSource,
      'transferFunction' : transferFunctionSource,
      'rayCast' : rayCastSource,
    })

    if False:
      print(self.shaderComputation.GetFragmentShaderSource())
      fp = open('/tmp/shader.glsl','w')
      fp.write(self.shaderComputation.GetFragmentShaderSource())
      fp.close()

    self.shaderComputation.SetTextureImageData(volumeToRender.GetImageData())

    resultImage = vtk.vtkImageData()
    resultImage.SetDimensions(1024, 1024, 1)
    resultImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    self.shaderComputation.SetResultImageData(resultImage)

    self.shaderComputation.Compute()

    # TODO:
    """
    - refactor test to support re-use of self.shaderComputation
    - fix vtkOpenGLShaderComputation to set MTime of resultImage (point data and image data)
    - test render performance
    - debug raycaster
    - add features!
    """

    if not hasattr(self, "iv"):
      self.iv = vtk.vtkImageViewer()
      self.iv.SetColorLevel(128)
      self.iv.SetColorWindow(256)
    self.iv.SetInputData(resultImage)
    self.iv.Render()

    if not hasattr(self,"renderWindow"):
      layoutManager = slicer.app.layoutManager()
      threeDWidget = layoutManager.threeDWidget(0)
      threeDView = threeDWidget.threeDView()
      self.renderWindow = threeDView.renderWindow()
      print('adding render observer')
      self.renderTag = self.renderWindow.AddObserver(vtk.vtkCommand.EndEvent, self.test_ShaderComputation2)
    slicer.modules.ShaderComputationInstance.test = self
